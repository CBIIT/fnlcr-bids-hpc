/*The Parallel Hello World Program*/

#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
  char *topology_addr; // char
  int task;
  int ntasks;
  char *cuda_visible_devices; // char
   
  MPI_Init(&argc,&argv);

  topology_addr = getenv("SLURM_TOPOLOGY_ADDR");
  MPI_Comm_rank(MPI_COMM_WORLD, &task);
  MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
  cuda_visible_devices = getenv("CUDA_VISIBLE_DEVICES");
     
  printf("Hello from node %s, rank %d / %d (CUDA_VISIBLE_DEVICES=%s)\n",topology_addr,task,ntasks,cuda_visible_devices);
            
  MPI_Finalize();
}
