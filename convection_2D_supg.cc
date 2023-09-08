#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/generic_linear_algebra.h>

namespace LA
{
    #if defined(DEAL_II_WITH_PETSC) && !(defined(DEAL_II_WITH_TRILINOS) && defined(FORCE_USE_OF_TRILINOS))
    using namespace dealii::LinearAlgebraPETSc;
    #  define USE_PETSC_LA
    #elif defined(DEAL_II_WITH_TRILINOS)
    using namespace dealii::LinearAlgebraTrilinos;
    #else
    #  error DEAL_II_WITH_PETSC or DEAL_II_WITH_TRILINOS required
    #endif
}

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_minres.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/grid/manifold.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>

#include <deal.II/lac/sparse_direct.h>

#include <deal.II/base/tensor_function.h>
// #include <deal.II/base/tensor_deprecated.h>

#include <deal.II/lac/sparse_ilu.h>

#include <iostream>
#include <fstream>
#include <sstream>

namespace Navierstokes
{
    using namespace dealii;
    
    template <int dim>
    class StokesProblem
    {        
    private:
        MPI_Comm                                  mpi_communicator;
        double deltat = 0.001;
        double totaltime = 2;
        int meshrefinement = 0;
        int degree;
        parallel::distributed::Triangulation<dim> triangulation;
        LA::MPI::SparseMatrix                     vof_system_matrix;
        DoFHandler<dim>                           vof_dof_handler;
        FESystem<dim>                             fevof;
        LA::MPI::Vector                           lr_vof_solution;
        LA::MPI::Vector                           lo_vof_system_rhs, lo_initial_condition_vof; 
        AffineConstraints<double>                 vofconstraints;
        IndexSet                                  owned_partitioning_vof;
        IndexSet                                  relevant_partitioning_vof;
        ConditionalOStream                        pcout;
        TimerOutput                               computing_timer;
        
    public:
        void setup_stokessystem();
        void setup_vofsystem();
        void assemble_vofsystem();
        void solve_stokes();
        double compute_errors();
        void output_results (int);
        void timeloop();
        
        StokesProblem(int degreein)
        :
        mpi_communicator (MPI_COMM_WORLD),
        degree(degreein),
        triangulation (mpi_communicator),
        vof_dof_handler(triangulation),
        fevof(FE_Q<dim>(degree), 1),
        pcout (std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)),
        computing_timer (mpi_communicator, pcout, TimerOutput::summary, TimerOutput::wall_times)
        {      
            pcout << "stokes constructor success...."<< std::endl;
        }
    };
    //===========================================================
    template <int dim>
    class VofRightHandSide : public Function<dim>
    {
    public:
        VofRightHandSide() : Function<dim>(1) {}
        virtual double value(const Point<dim> &, const unsigned int component = 0) const override;
    };
    
    template <int dim>
    double VofRightHandSide<dim>::value(const Point<dim> &, const unsigned int) const
    {
        return 0.0;
    }
    //==========================================
    template <int dim>
    class InitialValues : public Function<dim>
    {
    public:
//         int fac = 100;
        InitialValues () : Function<dim>(1) {}
        virtual double value (const Point<dim> &p, const unsigned int component = 0) const;
    };
    
    template <int dim>
    double InitialValues<dim>::value (const Point<dim> &p, const unsigned int) const
    {
        if(p[0] > 1 && p[0] < 2 && p[1] < 1.25 && p[1] > 0.25)
            return 1;
        else
            return 0;        
    }
    //==============================================================  
    template <int dim>
    void StokesProblem<dim>::setup_stokessystem()
    {  
        TimerOutput::Scope t(computing_timer, "setup_stokessystem");
        pcout <<"in setup_stokessystem "<<std::endl;
        GridIn<dim> grid_in;
        grid_in.attach_triangulation(triangulation);
        std::ifstream input_file("vof_2D_hex.msh");
        grid_in.read_msh(input_file);
        triangulation.refine_global (meshrefinement);
    }  
    //========================================================  
    template <int dim>
    void StokesProblem<dim>::setup_vofsystem()
    { 
        TimerOutput::Scope t(computing_timer, "setup_vofsystem");
        pcout <<"in setup_vofsystem "<<std::endl;
        vof_dof_handler.distribute_dofs(fevof);
        owned_partitioning_vof = vof_dof_handler.locally_owned_dofs();
        DoFTools::extract_locally_relevant_dofs (vof_dof_handler, relevant_partitioning_vof);
        
        {
            vofconstraints.clear();
            vofconstraints.reinit(relevant_partitioning_vof);
            //             DoFTools::make_hanging_node_constraints (vof_dof_handler, vofconstraints);
            //             std::set<types::boundary_id> no_normal_flux_boundaries;
            //             no_normal_flux_boundaries.insert (101);
            //             no_normal_flux_boundaries.insert (102);
            //             VectorTools::compute_no_normal_flux_constraints (move_dof_handler, 0, no_normal_flux_boundaries, moveconstraints);
            //             VectorTools::interpolate_boundary_values (move_dof_handler, 104, ZeroFunction<dim>(dim), moveconstraints);
            vofconstraints.close();
        }
        pcout << "Number of vof degrees of freedom: " << vof_dof_handler.n_dofs() << std::endl; 
        vof_system_matrix.clear();
        DynamicSparsityPattern vof_dsp(relevant_partitioning_vof);
        DoFTools::make_sparsity_pattern(vof_dof_handler, vof_dsp, vofconstraints, false);
        SparsityTools::distribute_sparsity_pattern (vof_dsp, vof_dof_handler.locally_owned_dofs(), mpi_communicator, relevant_partitioning_vof);        
        vof_system_matrix.reinit(owned_partitioning_vof, owned_partitioning_vof, vof_dsp, mpi_communicator);
        
        lr_vof_solution.reinit(owned_partitioning_vof, relevant_partitioning_vof, mpi_communicator);
        lo_vof_system_rhs.reinit(owned_partitioning_vof, mpi_communicator);
        lo_initial_condition_vof.reinit(owned_partitioning_vof, mpi_communicator);
        
        InitialValues<dim> initialcondition;
        VectorTools::interpolate(vof_dof_handler, initialcondition, lo_initial_condition_vof);
        lr_vof_solution = lo_initial_condition_vof;
        pcout <<"end of setup_vofsystem"<<std::endl;
    } 
    //=======================================
    template <int dim>
    void StokesProblem<dim>::assemble_vofsystem()
    {
        TimerOutput::Scope t(computing_timer, "assembly_vofsystem");
        pcout << "in assemble_vofsystem" << std::endl;
        vof_system_matrix=0;
        lo_vof_system_rhs=0;
        
        QGauss<dim>   vof_quadrature_formula(degree+2);
        
        FEValues<dim> fe_vof_values (fevof, vof_quadrature_formula,
                                     update_values  |
                                     update_quadrature_points |
                                     update_JxW_values |
                                     update_gradients);
        
        const unsigned int dofs_per_cell = fevof.dofs_per_cell;
        const unsigned int vof_n_q_points = vof_quadrature_formula.size();
        
        FullMatrix<double>                   vof_local_matrix (dofs_per_cell, dofs_per_cell);
        Vector<double>                       vof_local_rhs(dofs_per_cell);    
        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
        const VofRightHandSide<dim>          vof_right_hand_side;
        std::vector<double>                  vof_rhs_values(vof_n_q_points);
        std::vector<double>                  old_vof_values(vof_n_q_points);
        std::vector<double>                  value_phi_vof(dofs_per_cell);
        std::vector<Tensor<1,dim>>           gradient_phi_vof(dofs_per_cell);
        double deltak = 0.0;
                Tensor<1,dim> nsvelocity_values;
                nsvelocity_values[0] = 0.5;
                nsvelocity_values[1] = 0.0;
        typename DoFHandler<dim>::active_cell_iterator vof_cell = vof_dof_handler.begin_active(), vof_endc = vof_dof_handler.end();

        pcout << "CFL is "<< vof_cell->diameter()/0.5 << std::endl;

        for (; vof_cell!=vof_endc; ++vof_cell)
        {
            if (vof_cell->is_locally_owned())
            {
                deltak = 8.0*vof_cell->diameter(); // 0 for naive discretization
//                 deltak *= 4.0;
//                 if(vof_cell == vof_dof_handler.begin_active())

                fe_vof_values.reinit(vof_cell);
                vof_local_matrix = 0;
                vof_local_rhs = 0;
                vof_right_hand_side.value_list(fe_vof_values.get_quadrature_points(), vof_rhs_values);
                fe_vof_values.get_function_values(lr_vof_solution, old_vof_values);
                
                for (unsigned int q_index=0; q_index<vof_n_q_points; ++q_index)
                {
                    for (unsigned int k=0; k<dofs_per_cell; ++k)
                    {                        
                        value_phi_vof[k]   = fe_vof_values.shape_value (k, q_index);
                        gradient_phi_vof[k]= fe_vof_values.shape_grad (k, q_index);
                    }
                    for (unsigned int i=0; i<dofs_per_cell; ++i)            
                    {
                        for (unsigned int j=0; j<dofs_per_cell; ++j)
                        {                             
                            vof_local_matrix(i,j) += (value_phi_vof[j]*(value_phi_vof[i] + deltak*nsvelocity_values*gradient_phi_vof[i]) - deltat*(value_phi_vof[j])*(nsvelocity_values*gradient_phi_vof[i]) + deltat*deltak*(nsvelocity_values*gradient_phi_vof[j])*(nsvelocity_values*gradient_phi_vof[i]))*fe_vof_values.JxW(q_index);
                        }
                        vof_local_rhs(i) += old_vof_values[q_index]*(value_phi_vof[i] + deltak*(nsvelocity_values*gradient_phi_vof[i]))*fe_vof_values.JxW(q_index);
                    }                
                } //end of quadrature points loop
                vof_cell->get_dof_indices(local_dof_indices);                
                vofconstraints.distribute_local_to_global(vof_local_matrix, vof_local_rhs, local_dof_indices, vof_system_matrix, lo_vof_system_rhs);     
            } //end of if vof_cell->is_locally_owned()
        } //end of cell loop
        vof_system_matrix.compress (VectorOperation::add);
        lo_vof_system_rhs.compress (VectorOperation::add);
        pcout << "end of assemble_vofsystem"<< std::endl;
    }
    //====================================================
    //     template <int dim>
    //     double StokesProblem<dim>::compute_errors()
    //     {        
    //         const ComponentSelectFunction<dim> pressure_mask(dim, dim+1);
    //         const ComponentSelectFunction<dim> velocity_mask(std::make_pair(0,dim), dim+1);        
    //         Vector<double> cellwise_errors(triangulation.n_active_cells());
    //         QGauss<dim> quadrature(4);
    //         VectorTools::integrate_difference (dof_handler, lr_nonlinear_residue, ZeroFunction<dim>(dim+1), cellwise_errors, quadrature, VectorTools::L2_norm, &velocity_mask);
    //         const double u_l2_error = VectorTools::compute_global_error(triangulation, cellwise_errors, VectorTools::L2_norm);
    //         return u_l2_error;
    //     }
    //===========================================================
    template <int dim>
        double StokesProblem<dim>::compute_errors()
        {      
            Vector<double> cellwise_errors(triangulation.n_active_cells());
            QGauss<dim> quadrature(4);
            VectorTools::integrate_difference(vof_dof_handler, lr_vof_solution, ZeroFunction<dim>(1), cellwise_errors, quadrature, VectorTools::L1_norm);
            const double total_quantity = VectorTools::compute_global_error(triangulation, cellwise_errors, VectorTools::L1_norm);
            return total_quantity;
        }
            //===========================================================
//     template <int dim>
//         double StokesProblem<dim>::compute_linfty()
//         {      
//             Vector<double> cellwise_errors(triangulation.n_active_cells());
//             QGauss<dim> quadrature(4);
//             VectorTools::integrate_difference(vof_dof_handler, lr_vof_solution, ZeroFunction<dim>(1), cellwise_errors, quadrature, VectorTools::Linfty_norm);
//             const double total_quantity = VectorTools::compute_global_error(triangulation, cellwise_errors, VectorTools::Linfty_norm);
//             return total_quantity;
//         }
    //================================================================
    template <int dim>
    void StokesProblem<dim>::solve_stokes()
    {
        pcout <<"in solve_stokes"<<std::endl;
        TimerOutput::Scope t(computing_timer, "solve");
        LA::MPI::Vector  distributed_solution_vof_adjusted (owned_partitioning_vof, mpi_communicator);
        
        SolverControl solver_control_vof (vof_dof_handler.n_dofs(), 1e-12);
        dealii::PETScWrappers::SparseDirectMUMPS solver_vof(solver_control_vof, mpi_communicator);
        
        assemble_vofsystem();
        solver_vof.solve (vof_system_matrix, distributed_solution_vof_adjusted, lo_vof_system_rhs);
        
//         for(unsigned int i = distributed_solution_vof_adjusted.local_range().first; i < distributed_solution_vof_adjusted.local_range().second; ++i)
//         {
//             if(distributed_solution_vof_adjusted(i) > 1.0)
//                 distributed_solution_vof_adjusted(i) = 1.0;
//             else if(distributed_solution_vof_adjusted(i) < 0.0)
//                 distributed_solution_vof_adjusted(i) = 0.0;
//         }
//         distributed_solution_vof_adjusted.compress(VectorOperation::insert);
        
        vofconstraints.distribute(distributed_solution_vof_adjusted);
        lr_vof_solution = distributed_solution_vof_adjusted;
        pcout<< "=====linfty_norm===== " << lr_vof_solution.linfty_norm() << std::endl;
        pcout <<"end of solve_stokes "<<std::endl;
    }
    //===================================================================
    template <int dim>
    void StokesProblem<dim>::output_results(int timestepnumber)
    {
        TimerOutput::Scope t(computing_timer, "output");
        
        DataOut<dim> data_out;
        data_out.attach_dof_handler (vof_dof_handler);
        
        Vector<float> subdomain (triangulation.n_active_cells());
        for (unsigned int i=0; i<subdomain.size(); ++i)
            subdomain(i) = triangulation.locally_owned_subdomain();
        data_out.add_data_vector (subdomain, "subdomain");
        data_out.add_data_vector(vof_dof_handler, lr_vof_solution, "vof");
        data_out.build_patches ();
        
        std::string filenamebase = "zfs2d-8supg";
        
        const std::string filename = (filenamebase + Utilities::int_to_string (timestepnumber, 3) + "." +Utilities::int_to_string(triangulation.locally_owned_subdomain(), 4));
        std::ofstream output ((filename + ".vtu").c_str());
        data_out.write_vtu (output);
        
        if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
        {
            std::vector<std::string> filenames;
            for (unsigned int i=0; i<Utilities::MPI::n_mpi_processes(mpi_communicator); ++i)
                filenames.push_back (filenamebase + Utilities::int_to_string (timestepnumber, 3) + "." + Utilities::int_to_string (i, 4) + ".vtu");
            
            std::ofstream master_output ((filenamebase + Utilities::int_to_string (timestepnumber, 3) + ".pvtu").c_str());
            data_out.write_pvtu_record (master_output, filenames);
        }
    }
    //==================================================================  
    template <int dim>
    void StokesProblem<dim>::timeloop()
    {      
        double timet = deltat;
        int timestepnumber=0;
        
        while(timet<totaltime)
        {  
            output_results(timestepnumber);
            pcout << "total scalar quantity = " << compute_errors() << std::endl;
            solve_stokes();
            pcout <<"timet "<<timet <<std::endl;                       
            timet+=deltat;
            timestepnumber++;
        } 
        output_results(timestepnumber);
    }
}  // end of namespace
//====================================================
int main (int argc, char *argv[])
{
    try
    {
        using namespace dealii;
        using namespace Navierstokes;        
        Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);        
        StokesProblem<2> flow_problem(1);
        flow_problem.setup_stokessystem();
        flow_problem.setup_vofsystem();
        flow_problem.timeloop();
    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl << std::endl
        << "----------------------------------------------------"
        << std::endl;
        std::cerr << "Exception on processing: " << std::endl
        << exc.what() << std::endl
        << "Aborting!" << std::endl
        << "----------------------------------------------------"
        << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << std::endl << std::endl
        << "----------------------------------------------------"
        << std::endl;
        std::cerr << "Unknown exception!" << std::endl
        << "Aborting!" << std::endl
        << "----------------------------------------------------"
        << std::endl;
        return 1;
    }    
    return 0;
}
