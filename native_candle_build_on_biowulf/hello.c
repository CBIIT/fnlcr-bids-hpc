/*The Parallel Hello World Program*/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>


int main(int argc, char **argv)
{
  int task;
  int ntasks;
  char *cuda_visible_devices; // char
   
  MPI_Init(&argc,&argv);

  char *topology_addr; // char
  topology_addr = getenv("SLURM_TOPOLOGY_ADDR");
  MPI_Comm_rank(MPI_COMM_WORLD, &task);
  MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
  cuda_visible_devices = getenv("CUDA_VISIBLE_DEVICES");
  
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);
     
  // get hostname
  char hostname[50] = {'\0'};
  gethostname(hostname, 49);

  printf("Hello from SLURM_TOPOLOGY_ADDR %s, MPI_Get_processor_name %s, hostname %s, rank %d / %d (CUDA_VISIBLE_DEVICES=%s)\n",topology_addr,processor_name,hostname,task,ntasks,cuda_visible_devices);
            
  MPI_Finalize();
}







// int main(int argc, char** argv) {
//      // Initialize the MPI environment
//      MPI_Init(NULL, NULL);

//      // Get the number of processes
//      int world_size;
//      MPI_Comm_size(MPI_COMM_WORLD, &world_size);

//      // Get the rank of the process
//      int world_rank;
//      MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

//      // Get the name of the processor
//      char processor_name[MPI_MAX_PROCESSOR_NAME];
//      int name_len;
//      MPI_Get_processor_name(processor_name, &name_len);

//      // get hostname
//      char hostname[50] = {'\0'};
//      gethostname(hostname, 49);
//      // get CUDA_VISIBLE_DEVICES
//      const char *cvd = getenv("CUDA_VISIBLE_DEVICES");
//      // Print off a hello world message
//      printf("Hello world from processor %s, rank %d / %d on %s (CUDA_VISIBLE_DEVICES=%s)\n",
//             processor_name, world_rank, world_size, hostname, cvd);
//      // Finalize the MPI environment.
//      MPI_Finalize();
// }