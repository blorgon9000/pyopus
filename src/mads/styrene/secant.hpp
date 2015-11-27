/*
To use the secant solver to find the root of a scalar function:
   (the parametric object E must have public function such as E->f(x),
      where x is the point at which evaluate f.)
   1- construct the solver :    solver = new secant<E>();   
   2- set the solver :    solver->set(unit, x0, x1);   //unit is usually the pointer *this, and x0 and x1 are two required initial points
   3- launch the solver : bool = solver->run();   //will return true is success, false if the solver failed
*/
#ifndef SECANT_H
#define SECANT_H

#include "defines.hpp"
using namespace std;


template <class E>
class secant {
private:
  double x_last, x_now, x_next;
  double f_last, f_now, error;
  int i;
  bool OK;
  E *unit;
	  
public:
  secant(){
   x_last=0;
   x_now=0;
   x_next=0;
   f_last=0;
   f_now=0;
   OK=false;
}
;
  void set(E* tmp, double x1, double x2)
{
   unit=tmp;
   x_last=x1;
   x_now=x2;
   OK=false;
}
  ;
  bool run(){
  // if(DEBUG) cout<<endl<<"begin solve secant";
   f_last = unit->f(x_last);
   for (i=1; i<MAX_ITER_SECANT; i++)
   {
      f_now = unit->f(x_now);
     // if(DEBUG) cout<<endl<<" x = "<<x_now<<"    f(x) = "<<f_now;
      x_next = x_now - (f_now*(x_now-x_last)/(f_now-f_last));
      if (fabs((x_next-x_now)/x_now)<=TOL_SECANT)
      {
         i=MAX_ITER_SECANT;
         OK=true;
      }
      else
      {
         x_last=x_now;
         f_last=f_now;
         x_now=x_next;
      }
   }
   return OK;
}
;
  ~secant(){}
};
#endif
