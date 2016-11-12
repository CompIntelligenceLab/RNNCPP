#include <iostream>
#include "typedefs.h"
#include "print_utils.h"
#include "model.h"

using namespace std;

//----------------------------------------------------------------------
void U::print(VF3D x, std::string msg /*""*/)
{
	cout << msg << ",  shape: (" << x.n_rows << ", " << x.n_cols << ", " << x.n_slices << ")" << endl;
}
//----------------------------------------------------------------------
void U::print(VF2D_F x, std::string msg /*""*/)
{
	int n_rows = (int) x[0].n_rows;
	int n_cols = (int) x[0].n_cols;

	string same_size = "same size";

	// check that all matrics are the same size
	if (x.n_rows > 1) {
		for (int i=1; i < x.n_rows; i++) {
			if (x[i].n_rows != n_rows || x[i].n_cols != n_cols)  {
				same_size = "different sizes";
				break;
			}
		}
	}

	cout << msg << ",  field size: " << x.n_rows << ", shape: (" << x[0].n_rows << ", " << x[0].n_cols << ")" 
	     << ", " << same_size << endl;
} 
//----------------------------------------------------------------------
void U::print(VF2D_F x, int val1, std::string msg)
{
	char buf[80];
	sprintf(buf, msg.c_str(), val1);

	int n_rows = (int) x[0].n_rows;
	int n_cols = (int) x[0].n_cols;

	string same_size = "same size";

	// check that all matrics are the same size
	if (x.n_rows > 1) {
		for (int i=1; i < x.n_rows; i++) {
			if (x[i].n_rows != n_rows || x[i].n_cols != n_cols)  {
				same_size = "different sizes";
				break;
			}
		}
	}

	cout << buf << ", field size: " << x.n_rows << ", shape: (" << x[0].n_rows << ", " << x[0].n_cols << ")" 
	     << ", " << same_size << endl;
}
//----------------------------------------------------------------------
void U::print(VF2D x, std::string msg /*""*/)
{
	cout << msg << ", shape: " << x.n_rows << ", " << x.n_cols << endl;
} 
//----------------------------------------------------------------------
void U::print(VF2D x, int val1, std::string msg)
{
	char buf[80];
	sprintf(buf, msg.c_str(), val1);
	cout << buf << ", shape: (" << x.n_rows << ", " << x.n_cols << ")" << endl;
}
//----------------------------------------------------------------------
void U::print(LOSS x, std::string msg /*""*/)
{
	cout << msg << ", shape: " << x.n_rows << endl;
} 
//----------------------------------------------------------------------
void U::print(VF1D x, std::string msg /*""*/)
{
	cout << msg << ", shape: " << x.n_rows << endl;
} 
//----------------------------------------------------------------------
void U::print(VF1D x, int val1, std::string msg)
{
	char buf[80];
	sprintf(buf, msg.c_str(), val1);
	cout << buf << ", shape: (" << x.n_rows << ")" << endl;
}
//----------------------------------------------------------------------
void U::print(VF1D_F x, int val1, std::string msg)
{
	char buf[80];
	sprintf(buf, msg.c_str(), val1);

	int n_rows = (int) x[0].n_rows;

	string same_size = "same size";

	// check that all matrics are the same size
	if (x.n_rows > 1) {
		for (int i=1; i < x.n_rows; i++) {
			if (x[i].n_rows != n_rows) {
				same_size = "different sizes";
				break;
			}
		}
	}

	cout << buf << ", field size: " << x.n_rows << ", shape: (" << x[0].n_rows <<  ")" 
	     << same_size << endl;
}
//----------------------------------------------------------------------
void U::print(VF1D_F x, std::string msg /*""*/)
{

	int n_rows = (int) x[0].n_rows;

	string same_size = "same size";

	// check that all matrics are the same size
	if (x.n_rows > 1) {
		for (int i=1; i < x.n_rows; i++) {
			if (x[i].n_rows != n_rows) {
				same_size = "different sizes";
				break;
			}
		}
	}

	cout << msg << ", field size: " << x.n_rows << ", shape: (" << x[0].n_rows <<  ")" 
	     << same_size << endl;
}
//----------------------------------------------------------------------
// Efficiency is not the purpose. 
void U::matmul(VF2D_F& prod, const VF2D& mat, const VF2D_F& vec)
{
	for (int b=0; b < vec.n_rows; b++) {
		prod(b) = mat * vec(b);
	}
}
//----------------------------------------------------------------------
// Efficiency is not the purpose. 
void U::matmul(VF2D_F& prod, const VF2D& mat, const VF2D_F& vec, int seq)
{
	for (int b=0; b < vec.n_rows; b++) {
		prod(b).col(seq) = mat * vec(b).col(seq);
	}
}
//----------------------------------------------------------------------
// Efficiency is not the purpose. 
void U::matmul(VF2D_F& prod, const VF2D& mat, const VF2D_F& vec, int from, int to)
{
	for (int b=0; b < vec.n_rows; b++) {
		prod(b).col(to) = mat * vec(b).col(from);
	}
}
//----------------------------------------------------------------------
void U::createMat(VF2D_F& mat, int nb_batch, int nb_rows, int nb_cols)
{
	//arma::field<arma::Mat<float> > m; m.set_size(3);
	//VF2D_F mm; mm.set_size(3);
	mat.set_size(nb_batch);

	for (int b=0; b < nb_batch; b++) {
		mat(b) = VF2D(nb_rows, nb_cols);
	}
}
//----------------------------------------------------------------------
void U::zeros(VF2D_F& mat) 
{
	for (int b=0; b < mat.n_rows; b++) {
		mat(b).zeros();
	}
}
//----------------------------------------------------------------------
void U::t(VF2D_F& mat, VF2D_F& transpose)
{
	VF2D_F transp(mat.n_rows);

	for (int b=0; b < mat.n_rows; b++) {
		transpose[b] = mat[b].t();
	}
}
//----------------------------------------------------------------------
void U::leftTriad(VF2D_F& prod, VF2D_F& a, VF2D_F& b, VF2D_F& c)
{
	for (int p=0; p < a.n_rows; p++) {
		prod[p] = (a(p) % b(p)) * c(p);
	}
}
//----------------------------------------------------------------------
void U::leftTriad(VF2D_F& prod, VF2D_F& a, VF2D_F& b, VF2D_F& c, int from, int to)
{
	for (int p=0; p < a.n_rows; p++) {
		prod(p).col(to) = (a(p).col(from) % b(p).col(from)) * c(p).col(from);
	}
}
//----------------------------------------------------------------------
void U::rightTriad(VF2D_F& prod, const VF2D& a, const VF2D_F& b, const VF2D_F& c)
{
	for (int p=0; p < b.n_rows; p++) {
		prod[p] = a * (b(p) % c(p));
	}
}
//----------------------------------------------------------------------
void U::rightTriad(VF2D_F& prod, const VF2D& a, const VF2D_F& b, const VF2D_F& c, int from, int to)
{
	// only called during backpropagation
	if (to <  0) return;

	for (int p=0; p < b.n_rows; p++) {
		prod(p).col(to) = a * (b(p).col(from) % c(p).col(from));    // ERROR 
	}

	#if 0
	printf("\nrightTriad, from, to= %d, %d\n", from, to);
	print(a, "a");
	print(b, "b");
	print(c, "c");
	print(prod, "prod");
	a.print("a");
	b.print("b");
	c.print("c");
	prod.print("prod");
	//exit(0);
	#endif
}
//----------------------------------------------------------------------
void U::printRecurrentLayerLoopInputs(Model *m)
{
	LAYERS layers = m->getLayers();
	printf("----------------------------------------\n");
	printf("Recurrent Layer Loop Inputs\n");
	for (int i=0; i < layers.size(); i++) {
		Layer* layer = layers[i]; 
		Connection* con = layer->getConnection();
		if (con) {
			layer->printSummary();
			//layer->getLoopInput().print("loop input");
			layer->getLoopInput()[0].raw_print(cout, "loop input");
		}
	}
}
//----------------------------------------------------------------------
void U::printInputs(Model *m)
{
	LAYERS layers = m->getLayers();
	printf("----------------------------------------\n");
	printf("==> Layer Inputs (input to activation function)\n");
	for (int i=0; i < layers.size(); i++) {
		Layer* layer = layers[i]; 
		layer->printSummary();
		//layer->getInputs().print("inputs");
		layer->getInputs()[0].raw_print(cout, "inputs");
		Connection* con = layer->getConnection();
	}
}
//----------------------------------------------------------------------
void U::printLayerInputs(Model *m)
{
	LAYERS layers = m->getLayers();
	printf("----------------------------------------\n");
	printf("==> Layer Inputs (input to to layer, before they are summed up\n");
	for (int l=0; l < layers.size(); l++) {
		Layer* layer = layers[l]; 
		layer->printSummary();
		std::vector<VF2D_F> inputs = layer->getLayerInputs();
		for (int i=0; i < inputs.size(); i++) {
			printf("layer input %d\n", i);
			//inputs[i].print("input ");
			inputs[i][0].raw_print(cout, "input ");
		}
		Connection* con = layer->getConnection();
		if (con) {
			//layer->getLoopInput().print("loop input");
			layer->getLoopInput()[0].raw_print(cout, "loop input");
		}
	}
}
//----------------------------------------------------------------------
void U::printLayerOutputs(Model *m)
{
	LAYERS layers = m->getLayers();
	printf("----------------------------------------\n");
	printf("==> Layer Outputs (output of activation function)\n");
	for (int i=0; i < layers.size(); i++) {
		Layer* layer = layers[i]; 
		layer->printSummary();
		//layer->getOutputs().print("Outputs");
		layer->getOutputs()[0].raw_print(cout, "Outputs");
	}
}
//----------------------------------------------------------------------
void U::printLayerBiases(Model *m)
{
	LAYERS layers = m->getLayers();
	printf("----------------------------------------\n");
	printf("==> Layer Outputs (output of activation function)\n");
	for (int i=0; i < layers.size(); i++) {
		Layer* layer = layers[i]; 
		layer->printSummary();
		layer->getBias().raw_print(cout, "Bias");
	}
}
//----------------------------------------------------------------------
void U::printWeights(Model* m)
{
	printf("-------------------------------------------------\n");
	printf("==> PRINT WEIGHTS\n");
	CONNECTIONS cons = m->getConnections();
	printf("   Non-recurrent connections\n");
	for (int i=0; i < cons.size(); i++) {
		Connection* con = cons[i];
		con->printSummary("");
		con->getWeight().print("weight");
	}
	printf("\n   Recurrent connections\n");
	LAYERS layers = m->getLayers();
	for (int i=0; i < layers.size(); i++) {
		Layer* layer = layers[i]; 
		Connection* con = layer->getConnection();
		if (con) {
			con->printSummary("");
			con->getWeight().print("weight");
		}
	}
}
//----------------------------------------------------------------------
