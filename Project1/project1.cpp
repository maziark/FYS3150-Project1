#include <iostream>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <cstdlib>
#include <string>
#include <time.h>
#include <armadillo>
#include "time.h"

using namespace std;
using namespace arma;

#define TYPE double

TYPE find_f(TYPE x, TYPE h){
	return (h * h * 100.0 * exp(-10.0 * x));
}

TYPE find_u(TYPE x, TYPE h){
	//return (h * h * 100.0 * exp(-10.0 * x));
	return (1 - (1 - exp(-10.0)) * x - exp(-10.0*x));
}


void solve_matrix_equation_general	 (int N, TYPE *x, TYPE *u, TYPE *a, TYPE *b, TYPE *c);
void solve_matrix_equation		 (int N, TYPE *x, TYPE *u);
void solve_LU				 (int N, TYPE *x, TYPE *u);
TYPE CPU_runtime			 (void (*calculate) (int, TYPE *, TYPE *), int N, TYPE *x, TYPE *u, string method_name);
TYPE find_max_error 			 (int N, TYPE *x, TYPE *u);
void make_output_file			 (int N, TYPE *x, TYPE *u);
void make_output_stat			 (int k, TYPE *err, TYPE *cpu_runtime);
int main (int argc, char* argv[]){
	int power_of_ten = 3;

	// Statistics :
	TYPE *cpu_runtime	= new TYPE [2*power_of_ten];
	TYPE *err		= new TYPE [2*power_of_ten];


	for (int k = 1; k < 1 + power_of_ten; k++){
		
		
		int N			= pow(10.0, k*1.0);
		TYPE h			= 1.0/(N+1);
		TYPE *x			= new TYPE [N+2];
		TYPE *u_matrix		= new TYPE [N+2];
		TYPE *u_LU		= new TYPE [N+2];


		//initializing matrix x: 
	
		for (int i = 0; i < N+2; i++){
			x[i] = h * i;
			u_matrix[i] = 0;
			u_LU	[i] = 0;
			//x[i] = h*i;
		}
	
		cpu_runtime [2*(k-1)] 	= CPU_runtime (
			solve_matrix_equation, N, x, u_matrix, "Optimized version"
		);
	
		cpu_runtime [2*k-1] 	= CPU_runtime (
			solve_LU, N, x, u_matrix, "LU version"
		);
	
		err [2*(k-1)] 	= find_max_error (N, x, u_matrix);
		err [2*k-1] 	= find_max_error (N, x, u_LU);
	
		make_output_file (N, x, u_matrix);
			
	
		delete [] u_matrix; delete [] u_LU; delete [] x;
	}
	
	make_output_stat (power_of_ten, err, cpu_runtime);
	delete [] err; delete [] cpu_runtime;
	return 0;
}


TYPE CPU_runtime (void (*calculate) (int, TYPE *, TYPE *), int N, TYPE *x, TYPE *u, string method_name){
	clock_t start, finish;
	start = clock();
	calculate (N, x, u);
	finish = clock ();

	cout << "CPU runtime of method " << method_name << " " << ((finish - start)*1.0/CLOCKS_PER_SEC) << endl;
	return ( ((finish - start)*1.0/CLOCKS_PER_SEC));
}

void solve_LU (int N, TYPE * x, TYPE * u){
	mat A = zeros <mat>(N, N);
	vec f(N);
	mat L, U;
	vec X;
	vec temp;
	// initializing mat A:
	for (int i = 0; i < N; i++){
		for (int j = 0; j < N; j++){
			if (i == j) A(i, j) = 2;
			else if (abs (i - j) == 1) A(i, j) = -1;
		}
	}
	
	// initializing x : 
	
	TYPE h = 1.0/(N+1);
	for (int i = 0; i < N + 2; i++) x[i] = h*i;
	
	//initializing the value for function f:
	for (int i = 0; i < N+1; i++) f[i] = find_f (x[i], h);
	
	lu (L, U, A);
	X = solve (L, f);
	temp = solve (U, X);
	for (int i = 0; i < N; i++) u[i] = temp(i);
//	return solve (U,X);
}





void solve_matrix_equation (int N, TYPE *x, TYPE *u){
	//	Au = f; a, b, and c are  fields in A.
	TYPE h 	= 1.0/(N+1);

	TYPE *b	= new TYPE [N+2];
	
	TYPE *f = new TYPE [N+2];
	
	u [0]	= 0;
	u [N+1]	= 0;
	
	
	b[0] = 2;
	for (int i = 1; i < N+1; i++)
		b[i] = (i+1)*1.0/(1.0*i);
	

	//	initializing  f:
	
	for (int i = 0; i < N+1; i++)	f[i] = find_f(x[i], h);
	
	//	Forward subsitution :
	
	// Start from i = 1; as b[0] will remain the same; and so as f[1];
	for (int i = 1; i < N; i++)
		f[i] +=  (i-1)*1.0 * f[i - 1] / (1.0 * i) ;
		    
	//	Backward subsitution : 
	// Finding the last value of u, before doign the backward subsitution
	u[N] = f[N]/b[N];
	//for (int i = N-1; i > 0; i--)	u[i] = (f[i] + u[i+1]) / b[i]; 
	for (int i = N-1; i > 0; i--)	u[i] = i*1.0/(1.0*(i+1.0)) *(f[i] + u[i+1]); 
//	for (int i = 0; i < N+2; i++ ) cout << u[i] << "\t" << find_u(x[i], h)<<endl;
//	cout << x[0] << " " << x[N]<< endl;
	delete [] b; delete [] f;
	
}

TYPE find_max_error 			 (int N, TYPE *x, TYPE *u){
	TYPE max_error = -99999;
	TYPE h = 1.0/(N+1);
	for (int i = 1; i < N; i++){
		TYPE temp = log10 (abs(u[i]-find_u(x[i], h))/find_u(x[i], h));
		//cout << temp << endl;
		if (max_error < temp) max_error = temp;
	}
	return max_error;
}


void make_output_file (int N, TYPE *x, TYPE *u){
	string file_name = "output_files/result_" + to_string(N) + ".out";
	TYPE h = 1/(N+1);
	ofstream output (file_name);
	
	if (output.is_open()){
		for (int i = 1; i < N; i++)
			output << (find_u(x[i], h) - u[i]) << endl;
		
		output.close();
	}
	
	cout << "result saved for " + to_string (N) << endl;
}

void make_output_stat			 (int k, TYPE *err, TYPE *cpu_runtime){
	string file_name = "output_files/statistics.out";
	ofstream output (file_name);
	
	if (output.is_open()){
		for (int i = 0; i < k; i++){
			string temp = to_string (i+1) + "\t" + to_string(cpu_runtime[2*i]) + "\t" + to_string(cpu_runtime[2*i+1]) + "\t" 
				+ to_string(err[2*i]) + "\t" + to_string(err[2*i+1]);
			output << temp << endl;
		}
		output.close();
	}
	
	cout << "Saved stats" << endl;
}