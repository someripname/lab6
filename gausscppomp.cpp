#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
using namespace std;

int main(int argc, char**argv)
{
int i,j,k,num,numproc;
double tmp,t1,t2,t3,t4;

tmp=t1=t2=t3=t4=0;
num=numproc=0;

if (argc!=3)
 cout<<"\n !!!ERROR!!! Example of command: ./gauss <dimension of LSAE>
<number of processor cores> !!!ERROR!!!\n"<<endl, exit(1);

cout<<"**********************************************************
**************"<<endl;
 num=atoi(*++argv); // Enter the size LSAE
 numproc=atoi(*++argv); // Enter the number of processors
 cout<<"\tDimension of LSAE = "<<num<<"\tNumber of processor cores =
"<<numproc<<endl;

double **massiv = new double * [num];
 for (i=0; i<num; i++)
 massiv[i] = new double [num+1];
double *xi = new double [num];
double *x = new double [num];

t3=omp_get_wtime();
 for (i=0;i<num;i++)
 for(j=0;j<num;j++)
 while ((massiv[i][j]=5.-floor(rand()/3276.7+.5))==0);

#pragma omp parallel for private (j) shared (xi)
 for (j=0;j<num;j++)
 while ((xi[j]=5.-floor(rand()/3276.7+.5))==0);

 t4=omp_get_wtime();
 cout<<"\n\t\t\tRuntime 1 = "<<floor((t4-t3)*100.+0.5)/100.<<"(s)"<<endl;

 for (i=0;i<num;i++)
 for(j=0;j<num;j++)
 massiv[i][num]+=massiv[i][j]*xi[j];

 omp_set_num_threads(numproc);
 t1=omp_get_wtime();
 for (k=0; k<num; k++)
 {
 tmp=abs(massiv[k][k]), i=k;
#pragma omp parallel for private (j) shared (massiv)
 for (j=k+1; j<num; j++)
 if(abs(massiv[j][k])>tmp)
 i=j, tmp=abs(massiv[j][k]);

 if (i!=k)
 {
#pragma omp parallel for private (j,tmp) shared (massiv)
 for (j=k; j<=num; j++)
 {
 tmp=massiv[k][j];
 massiv[k][j]=massiv[i][j];
 massiv[i][j]=tmp;
 }
 }

#pragma omp parallel for private (j) shared (massiv)
 for (j=num; j>=k+1; j--)
 massiv[k][j]/=massiv[k][k];

#pragma omp parallel for private (i,j,tmp) shared (massiv)
 for (i=k+1; i<num; i++)
 {
 tmp=massiv[i][k];
 massiv[i][k]=0;
 if (tmp!=0)
 for (j=k+1; j<num+1; j++)
 massiv[i][j]-=tmp*massiv[k][j];
 }
 }

for(j=num-1; j>=0; j--)
 {
 *(x+j)=massiv[j][num];
#pragma omp parallel for private (i)
 for (i=j; i>=0; i--)
 massiv[i][num]-=massiv[i][j]**(x+j);
 }
 t2=omp_get_wtime();
 cout<<"\n\t\t\tRuntime = "<<floor((t2-t1)*100.+0.5)/100.<<"(s)"<<endl;
 cout<<"\nCorrect answer:"<<endl;
 for (i=0; i<10; i++)
 cout<<" xi"<<i+1<<"="<<setw(2)<<*(xi+i);
 cout<<endl;
 cout<<"Solution:"<<endl;
 for (i=0; i<10; i++)
 cout<<" x"<<i+1<<"="<<setw(2)<<*(x+i);
 cout<<endl;

cout<<"**********************************************************
**************"<<endl;
for (i=0; i<num+1; i++)
 delete [] massiv[i];
delete [] massiv;
 delete [] xi;
 delete [] x;
return 0;
}
