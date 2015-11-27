/*
This unit simulates a chemical reactor. Actually, only the pfr can be used.
(ref : Fogler).

Structure in the .process file:
reactor {name} {pfr or cstr} {index of input stream} {index of output stream} {length in m} {diameter in m} {nb_react, list of reactions occuring} {U in kW/m2.K}{Ta in K}

How to use:
   1- Call the constructor : react = new reactor<pfr or cstr>(in, out);		
   2- Set dimensions and reactions : react->set(length, diameter, nb_react, list_react);	//list_react is the list of reactions names
   3- Set cooling parameters : react->set(U, Ta);
   4- Set the name : react->set(name);
   5- Run the model: react->solve();
*/
#ifndef REACTOR_H
#define REACTOR_H

#include "pfr.hpp"
using namespace std;

template<class TYPE>
class reactor {
private:
  // ofstream log;
  bool success;
  string name;
  int i ,j, m, n;
  double V, L, D, U, Ta;
  stream *in, *out;
  TYPE *model;
  reaction ** rx;
  double ** table;

public:
  // reactor(){};
  reactor(stream* s1, stream* s2)
{
   in  = s1;
   out = s2;

   model = NULL;
}
  ;
  void set( const string & n) { name = n; }
  void set( double l , double d , int nb , const string * list_rx ) {
  m      = in->nb;
  n      = nb;
  L      = l;
  D      = d;
  V      = pi*pow(D/2.0, 2)*L;

  double * yields = new double [n];

  rx = new reaction * [n];
  for ( j = 0 ; j < n ; j++ )
    rx[j] = new reaction ( list_rx[j] , m , in->chem );

  table = new double * [m];
  for ( i = 0 ; i < m ; i++ )
    table[i] = new double[n];
  for ( j = 0 ; j < n ; j++ )
    for ( i = 0 ; i < m ; i++ )
      table[i][j] = rx[j]->a[i];
  for ( j = 0 ; j < n ; j++ ) {
    yields[j]=0.0;
    for ( i = 0 ; i < m ; i++ )
      if ( table[i][j] < 0 ) {
	yields[j]=in->chem[i]->n();
	i=m;
      }
  }

  delete [] yields;

}
;
  void set(double u, double ta) {U=u;Ta=ta;}
  bool solve(){

  if (model)
    delete model;
  model = new TYPE(in, out, table, n, rx, U, Ta);
  model->set(name);
  model->set(L,D);
  
  success = model->run();

  //    if(fabs(in->m-out->m)>sqrt(EPS))
  //    {
  //       log.open(MESSAGES, ios::app);
  //       log<<"   --> Warning <--  Block "<<name<<" is not in mass balance ("<<fabs(in->m-out->m)/in->m<<").\n";
  //       log.close();
  //    }

  
  // out->write(); // WRITE TOTO

  return success;
}
;
  void write(){

  cout << setprecision(6);

  cout << "WRITE FILE " << RUNTIME << name << ".unit" << " :\n\tBEGIN\n";
  cout << "\t>>         " << name;
  cout << endl << "\t>>           stream in : " << in->name;
  cout << endl << "\t>>           stream out : " << out->name;
  cout << endl << "\t>>           P = " << in->P
       << " atm,  T(in) = " << in->T << ",  T(out) =  " << out->T << "  K";
  cout << endl << "\t>>           L = " << L << ",  D = " << D << "  m";
  if (success)
    cout << " (converge normally)";
  cout << "\n\tEND\n\n";
  model->cost();
  model->water();
}
;

  double get_cost  ( void ) const { return model->get_cost() ; }
  double get_water ( void ) const { return model->get_water(); }

  ~reactor(){
  for ( i = 0 ; i < n ; i++ )
    delete rx[i];
  delete [] rx;
  for ( i = 0 ; i < m ; i++ )
    delete table[i];
  delete [] table;

  if (model)
    delete model;
}
;
};
#endif
