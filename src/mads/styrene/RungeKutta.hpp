/*
To use the Runge-Kutta solver for systems of first order differential equations,
   (the parametric object E must have public function such as E->f(i,x,y),
      where i is the index of the function to evaluate, x is the time and
	  y is a point such as y(x), returns values of differential equation i)
	  
   1- construct the solver :    solver = new RungeKutta<objet>(int);   //the integer is the dimension of x
   2- set the solver :    solver->set(unit, y0, x0, xn);   //unit is usually the pointer *this, y0 are the initial conditions, and x0 and xn is the time interval
   3- launch the solver : bool = solver->run();   //will return true is success, false if the solver failed????????????????????????????
   4- delete the solver : delete solver;
(ref  :Fortin)
*/
#ifndef RUNGEKUTTA_H
#define RUNGEKUTTA_H

#include "defines.hpp"
using namespace std;

template <class E>
class RungeKutta
{
private:
  double *k1, *k2, *k3, *k4, *y_tmp, *y;
  // double k1[MAX_DIM], k2[MAX_DIM], k3[MAX_DIM], k4[MAX_DIM], y_tmp[MAX_DIM], y[MAX_DIM];
  double h, x0, xn, x;
  int i, j, m;
  bool success;
  E *unit;
	  
public:
  RungeKutta(int dim){
   m = dim;
   k1 = new double[m];
   k2 = new double[m];
   k3 = new double[m];
   k4 = new double[m];
   y = new double[m];
   y_tmp = new double[m];
}
;
  ~RungeKutta(){
   delete [] k1;
   delete [] k2;
   delete [] k3;
   delete [] k4;
   delete [] y;
   delete [] y_tmp;
}
;
  void set( E * tmp , double * y0 , double beg , double end )
{
  unit=tmp;
  x0=beg; xn=end;
  x=x0;
  h=double(xn-x0)/double(N_INTER);
  for (i=0;i<m;i++) {y[i]=y0[i];}
  success=true;
}
  ;
  double dx(){return h;}
  bool run(){
  for(j=0;j<MAX_ITER_RK;j++) {
    //Avoid going out of x interval
    if (x+h >xn) {
      h = xn-x;
      j = MAX_ITER_RK;
    }

    //Compute k1, k2, k3, k4
    for(i=0;i<m;i++)
      k1[i] = h*unit->f(i, x, y);
    for(i=0;i<m;i++)
      y_tmp[i] = y[i]+k1[i]/2.0;
    for(i=0;i<m;i++)
      k2[i] = h*unit->f(i, x+h/2.0, y_tmp);
    for(i=0;i<m;i++)
      y_tmp[i] = y[i]+k2[i]/2.0;
    for(i=0;i<m;i++)
      k3[i] = h*unit->f(i, x+h/2.0, y_tmp);
    for(i=0;i<m;i++)
      y_tmp[i] = y[i]+k3[i];
    for ( i = 0 ; i < m ; i++ )
      k4[i] = h*unit->f ( i , x+h , y_tmp );
    //Compute the new y
    for(i=0;i<m;i++)
      y[i]+=(k1[i]+2*k2[i]+2*k3[i]+k4[i])/6.0;
    x += h;
  }

  if ( x < xn-EPS ) {// MODIF SEB (le EPS)
    success=false;
    
    // cout.setf(ios::fixed);
    // cout << setprecision(12);
    // cout << "x=" << x << " < xn=" << xn << " diff=" << xn-x << endl;
  }

  return success;
}
;
};
#endif
