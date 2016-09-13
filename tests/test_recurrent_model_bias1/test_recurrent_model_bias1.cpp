#include <math.h>
#include "../common.h"
#include <string>
#include <cstdlib>

float derivLoss(int k, VF1D& z0, VF1D& e, VF2D& w11, float alpha, int m0, int n0, std::vector<VF2D>& ws);
float predict(int k, VF1D& z0, VF1D& e, VF2D& w11, float alpha, int m0, int n0, std::vector<VF2D>& ws);

//void testRecurrentModelBias1(int nb_batch, int seq_len, int layer_size, int is_recurrent, Activation* activation, 
void testRecurrentModelBias1(Model* m, int layer_size, int is_recurrent, Activation* activation) 
{
	printf("\n\n\n");
	printf("=============== BEGIN test_recurrent_model_bias2  =======================\n");

	//================================
	int seq_len = m->getSeqLen();
	int nb_batch = m->getBatchSize();
	int input_dim  = 1;
	//m->setSeqLen(seq_len); // runs (but who knows whether correct) with seq_len > 1

	// I am not sure that batchSize and nb_batch are the same thing
	//m->setBatchSize(nb_batch);

	// Layers automatically adjust ther input_dim to match the output_dim of the previous layer
	// 2 is the dimensionality of the data
	// the names have a counter value attached to it, so there is no duplication. 
	Layer* input = new InputLayer(input_dim, "input_layer");

	Layer *d1;

	if (is_recurrent) {
		d1    = new RecurrentLayer(layer_size, "rdense");
	} else {
		d1    = new DenseLayer(layer_size, "rdense");
	}

	m->add(0,     input);
	m->add(input, d1);

	input->setActivation(activation); 
       d1->setActivation(activation); 

	m->addInputLayer(input);
	m->addOutputLayer(d1);

	m->printSummary();
	m->connectionOrderClean(); // no print statements

	m->initializeWeights();

	//===========================================
/***
Check the sequences: prediction and back prop.

1) dimension = 1, identity activation functions
   seq=2

 l=0    l=1    l=2
  In --> d1 --> loss0    (t=0)
         |  
         | 
         v
  In --> d1 --> loss1    (t=1)

Matrix form: 

y1(n) = f(w01 * x0(n) + w11 * y1(n-1))
Assume only a single x is input: then, 
y1(n) = w11^2 y1(n-2) = w11^n y1(0)
                      = w11^n (w01 * x0(0))
dL/dw11(i,j) = dL/da(k) * d/dw11(i,j) (n * W11^(n-1)) dW11/dw11(i,j) * (w01.col(0)*x0(0))
dAw/dw(i,j) = d/dw(i,j) (A(p,q) w(q,l)) = A(p,q) delta(i,q) delta(j,l)
                                        = A(p,i) delta(j,l)
Loss(n) = Loss(y1(n))
w01(nn,1), w11(nn,nn), x0(1), x1(nn)
nn = layer_size)


Inputs to nodes: z(l,t), a(l,t-1)
Output to nodes: a(l,t)
Weights: In -- d1 : w1
Weights: d1 -- d1 : w11
d1: l=1
exact(t): exact results at time t
loss0(a(2,0), exact(0))
loss1(a(2,1), exact(1))
Input at t=0: x0
Input at t=1: x1

Loss = L = loss0 + loss1
	z(0,0) = x0;
	a(0,0) = z(0,0);
	//a(1,-1) is retrieved via  layer->getConnection()->from->getOutputs();
	z(1,0) = w01  * a(0,0); // + w11 * a(1,-1); // a(1,-1) is initially zero
	a(1,0) = z(1,0);

	z(0,1) = x1;
	a(0,1) = z(0,1); // input <-- output  (set input to output)
	z(1,1) = w01  * a(0,1) + w11 * a(1,0);   // for delay of 2, the 2nd term would be: w11 * a(1,-1)
	a(1,1) = z(1,1);

Forward: 
 a(1,-1) = 0, z(1,-1) = w11 * a(1,-1)
 a(2,-1) = 0, z(1,-1) = w22 * a(2,-1)
 a(1,0) = z(1,0) = w1*x0     + w12 * z(1,-1)
 a(2,0) = z(2,0) = w2*z(1,0) + w12 * z(2,-1)
 -------
 z(1,0) = w11 * a(1,0)
 z(2,0) = w22 * a(2,0)
 a(1,1) = z(2,1) = w1*x1     + w12 * z(1,0)
 a(2,1) = z(2,1) = w2*z(1,1) + w12 * z(2,0)
***/

	float w01 = .4;
	float w11 = .6;
	float bias1 = -.7;  // single layer of size 1 ==> single bias
	float bias2 = -.45; // single layer of size 1 ==> single bias
	//w01       = 1.;
	//w11       = 1.;
	//w22       = 1.;
	float x0       = .45;
	float x1       = .75;
	float ex0      = .75; // exact value
	float ex1      = .85; // exact value
	VF2D a(layer_size+1, seq_len); // assume al dimensions = 1
	VF2D z(layer_size+1, seq_len); // assume al dimensions = 1

#if 0
	// t=0
	z(0,0) = x0;
	a(0,0) = z(0,0);
	//a(1,-1) is retrieved via  layer->getConnection()->from->getOutputs();
	z(1,0) = w01 * a(0,0) + bias1; // + w11 * a(1,-1); // a(1,-1) is initially zero
	a(1,0) = (z(1,0));
	a.print("a");
	z(2,0) = w12 * a(1,0) + bias2;
	exit(0);
	a(2,0) = (z(2,0));

	// t=1
	z(0,1) = x1;
	a(0,1) = z(0,1);
	z(1,1) = w01 * a(0,1) + w11 * a(1,0) + bias1;
	a(1,1) = (z(1,1));
	z(2,1) = w12 * a(1,1) + w22 * a(2,0) + bias2;
	a(2,1) = (z(2,1));

	// loss = loss0 + loss1
	//      = (a(2,0)-ex0)^2 + (a(2,1)-ex1)^2
	//      = (w12*a(1,0)-ex0)^2 + (w12*a(1,1)+w22*a(2,0)-ex1)^2
	//      = (w12*w01*a(0,0)-ex0)^2 + (w12*(w01*a(0,1)+w11*a(1,0)) + w22*w12*w01*x0 -ex1)^2
	//      = (w12*w01*x0-ex0)^2 + (w12*(w01*x1+w11*w01*x0) + w22*w12*w01*x0 -ex1)^2
	//      = (l1-ex0)^2 + (l2+l3-ex1)^2
	//
	// d(loss)/dw1  = 2*(l1-ex0)*w12*x0 + 2*(l2+l3-ex1)*(w12*x1) + 2*(l2+l3-ex1)*w22*w12*x0
	// d(loss)/dw12 = 2*(l1-ex0)*w01*x0 + 2*(l2+l3-ex1)*(w01*x1+w11*w01*x0) + 2*(l2+l3-ex1)*w22*w01*x0
	// d(loss)/dw11 = 2*(l2+l3-ex1)*(w12*w01*x0) 
	// d(loss)/dw22 = 2*(l2+l3-ex1)*(w12*w01*x0) 
	//

	float loss0 = (a(2,0)-ex0)*(a(2,0)-ex0);  // same as predict routine
	float loss1 = (a(2,1)-ex1)*(a(2,1)-ex1);  // same as predict routine
	float L0 = 2.*(a(2,0)-ex0);
	float L1 = 2.*(a(2,1)-ex1);
	float dldw1  = L0*w12*x0 + L1*(w12*x1+w12*w11*x0+w22*w12*x0); // CORRECT
	float dldw12 = L0*w01*x0 + L1*(w01*x1+w11*w01*x0 + w22*w01*x0); // CORRECT
	float dldw11 = L1*(w12*w01*x0); // CORRECT
	float dldw22 = L1*(w12*w01*x0); // CORRECT


	#if 0
	printf("\n ============== Layer Outputs =======================\n");
	printf("a00, a01= %f, %f\n", a(0,0), a(0,1));
	printf("z10, z11= %f, %f\n", z(1,0), z(1,1));
	printf("a10, a11= %f, %f\n", a(1,0), a(1,1));
	printf("z20, z21= %f, %f\n", z(2,0), z(2,1));
	printf("a20, a21= %f, %f\n", a(2,0), a(2,1));
	#endif

	//a11 = w01 *a01 + w11*a10;
	//a21 = w12*a11 + w22*a20;
	//a10 = w01*a00
	//a20 = w01*w11*a10;

	float da11da01 = w01;
	float da11da10 = w11;
	float da21da11 = w12;
	float da21da20 = w22;
	float da10da00 = w01;
	float da20da10 = w01*w11;

	#if 0
	printf("dLda20= %f\n", L0);
	printf("dLda21= %f\n", L1);
	printf("da11da01= %f\n", w01);
	printf("da11da10= %f\n", w11);
	printf("da21da11= %f\n", w12);
	printf("da21da20= %f\n", w22);
	printf("da10da00= %f\n", w01);
	printf("da20da10= %f\n", w01*w11);
	printf("\n\n");
	#endif

	float dLda20 = L0;
	float dLda21 = L1;
	float dLda11 = L1 * da21da11;
	float dLda10 = L0*w01*w11;
	float dLda01 = w01*w12*L1;
	float dLda00 = w01*(L1*w11*w12 + L0*w01*w11);
	//float dLda00 = da10da00*(L1*da11da10*da21d11 + L0*da20da10) = w01*(L1*w11*w12 + L0*w01*w11);

	#if 0
	printf("Calculation of weight derivatives by hand\n");
	printf("loss0= %f, loss1= %f\n", loss0, loss1);
	printf("dldw1= %f\n", dldw1);
	printf("dldw12= %f\n", dldw12);
	printf("dldw11= %f\n", dldw11);
	printf("dldw22= %f\n", dldw22);
	printf("\n");

	printf("dLda by hand\n");
	printf("dLda20= %f\n", L0);
	printf("dLda21= %f\n", L1);
	printf("dLda11= %f\n", dLda11);
	printf("dLda10= %f\n", dLda10);
	printf("dLda01= %f\n", dLda01);
	printf("dLda00= %f\n", dLda00);
	#endif
#endif

	int output_dim = m->getOutputLayers()[0]->getOutputDim();

	VF2D_F xf(nb_batch), exact(nb_batch);
	for (int b=0; b < nb_batch; b++) {
		xf(b) = VF2D(input_dim, seq_len);
		for (int i=0; i < input_dim; i++) {
			for (int s=0; s < seq_len; s++) {
				if (s == 0) {
					xf(b)(i,s) = x0; 
				} else {
					xf(b)(i,s) = 0.;  // compare with analytics
				}
			}
		}
		printf("*** xf:   ONLY HAS INITIAL COMPONENT at t=0, else ZERO\n");

		exact(b) = VF2D(output_dim, seq_len);
		for (int i=0; i < output_dim; i++) {
			for (int s=0; s < seq_len; s++) {
				exact(b)(i,s) = ex0; 
			}
		}
	}

	#if 1
	WEIGHT ww01;
	{
		Connection* conn;
		conn = m->getConnection(input, d1);
		WEIGHT& w_01 = conn->getWeight();

		//w_01.print("w01: initial weight\n");

		//w_01(0,0) = w01;
		conn->computeWeightTranspose();
		ww01 = w_01; // TEMPORARY
	}
	#endif

	#if 1
	{
		Connection* conn;
		conn = d1->getConnection();
		if (conn) {
			WEIGHT& w_11 = conn->getWeight();
			// transform w_11 to double
			//arma::Mat<double> d_11(size(w_11));
			arma::Mat<double> d_11 =  arma::conv_to<arma::Mat<double> >::from(w_11);
			//for (int i=0; i < d_11.size(); i++) {
				//d_11(i) = w_11(i);
			//}
			arma::cx_vec eigval = arma::eig_gen(d_11);
			//w_11.print("w11");
			//d_11.print("d11");
			arma::cx_vec ee = arma::sort(eigval, "descend");
			arma::Col<double> re = arma::real(ee);
			arma::Col<double> im = arma::imag(ee);
			arma::real(ee).print("real(ee)");
			arma::imag(ee).print("imag(ee)");
			//ee.print("eigenvalues");
			double max_eig = sqrt(re[0]*re[0] + im[0]*im[0]);
			//double max_eigen = sqrt(ee[0]*ee[0] + ee[1]*ee[1]);
			printf("\n>>>>> max_eigen= %f\n\n", max_eig);
			//w_11.print("w11: initial weight\n");
			//w_11(0,0) = w11;
			conn->computeWeightTranspose();

			// analytic calculation: 
			VF2D w = w_11;

			printf("Vector of pow(w,k)\n");
			std::vector<VF2D> ws; 
			ws.push_back(w.eye(size(w)));
			for (int s=0; s < seq_len; s++) {
				w = w * w_11;
				ws.push_back(w);
				printf("power = %d\n", s);
			}
			w.print("w power");
			ws[0].print("ws[0]");
			ws[1].print("ws[1]");
			ws[seq_len-1].print("ws[seq_len-1]");
			ws[seq_len].print("ws[seq_len]");
			//exit(0);

			// Compute exact derivatives for one weight element, for starters. 
			float alpha = w_11(0,0);
			//xf.print("xf"); 
			//printf("alpha= %f\n", alpha); 
			int nr = w_11.n_rows;
			int nc = w_11.n_cols;
			float alphap = pow(alpha, seq_len);
			//printf("seq_len= %d\n", seq_len);
			//printf("alphap= %f\n", alphap);
			exact(0).print("exact(0)");

			int k;
			//U::print(ww01, "ww01");
			//xf(0).col(0).print("xf(0).col(0)");
			//xf(0).col(1).print("xf(0).col(1)");
			//z0.print("z0");
			//VF1D y = xf;
			//e.print("e");
			
			VF1D z0 = ww01*xf(0).col(0); // ERROR
			VF2D total_deriv(size(w_11));

			nr = total_deriv.n_rows;
			nc = total_deriv.n_cols;
			nr = nr > 3 ? 3 : nr;
			nc = nc > 3 ? 3 : nc;

			// derivative of L(k) with respect to w_11(m0, n0)
			for (int m0=0; m0 < nr; m0++) {
			for (int n0=0; n0 < nc; n0++) {
				printf("---- m0,n0= %d, %d\n", m0, n0);
				printf("seq_len= %d\n", seq_len);
				total_deriv(m0,n0) = 0.;
				for (int k=0; k < seq_len; k++) {
					VF1D e = exact(0).col(k);  // exact can be arbitrary
					float dl = derivLoss(k, z0, e, w_11, alpha, m0, n0, ws);
					total_deriv(m0,n0) += dl;
					//printf("dl= %f\n", dl);
				}
				VF1D e = exact(0).col(seq_len-1);  // exact can be arbitrary
				predict(seq_len-1, z0, e, w_11, alpha, m0, n0, ws);
				printf("total_deriv(%d,%d)= %f\n", m0, n0, total_deriv(m0,n0));
			}}
			//exit(0);
		}
	}
	#endif
	//exit(0);
	
	#if 1
	{
		BIAS& bias_1 = d1->getBias();
		bias_1(0) = 0.; //bias1;
	}
	#endif

	//================================
	float inc = 0.001;
	runTest(m, inc, xf, exact); 
	exit(0);

	predictAndBackProp(m, xf, exact);

	exit(0);
}
//----------------------------------------------------------------------
float predict(int k, VF1D& z0, VF1D& e, VF2D& w11, float alpha, int m0, int n0, std::vector<VF2D>& ws)
{
	VF1D x = z0;
	printf("iteration: 0, x= %14.7f\n", x(0));  //x.print("x: ");

	for (int i=0; i < k; i++) {
		x = w11*x;
		printf("iteration: %d, x= %14.7f\n", i+1, x(0));  //x.print("x: ");
	}
}
//----------------------------------------------------------------------
float derivLoss(int k, VF1D& z0, VF1D& e, VF2D& w11, float alpha, int m0, int n0, std::vector<VF2D>& ws)
{
	VF2D x(1,1);
	VF1D l = 2.*(z0-e); //.t();
	float Lprime = l[m0] * k *pow(alpha, k-1) * z0[n0];
	return Lprime;
}
//----------------------------------------------------------------------
int main(int argc, char* argv[])
{
// arguments: -b nb_batch, -l layer_size, -s seq_len, -s is_recursive

    int nb_batch = 1;
    int layer_size = 1;
    int seq_len = 1;
    int is_recurrent = 1;
	Activation* activation = new Identity(); 
	std::string initialization_type;
	initialization_type = "xavier";

	argv++; 
	argc--; 

	printf("argc= %d\n", argc);
	while (argc > 0) {
		std::string arg = std::string(argv[0]);
		printf("arg= %s\n", arg.c_str());
		if (arg == "-b") {
			nb_batch = atoi(argv[1]);
			argc -= 2; argv += 2;
		} else if (arg == "-s") {
			seq_len = atoi(argv[1]);
			argc -= 2; argv += 2;
		} else if (arg == "-r") {
			is_recurrent = atoi(argv[1]);
			argc -= 2; argv += 2;
		} else if (arg == "-w") {
			initialization_type = argv[1];
			argc -= 2; argv += 2;
		printf("init type: %s\n", initialization_type.c_str());
		} else if (arg == "-l") {
			layer_size = atoi(argv[1]);
			argc -= 2; argv += 2;
		} else if (arg == "-a") {
			std::string name = argv[1];
			//printf("name= %s\n", name.c_str());
			if (name == "tanh") {
				activation = new Tanh();
			} else if (name == "iden") {
				activation = new Identity();
			} else if (name == "sigmoid") {
				activation = new Sigmoid();
			} else if (name == "relu") {
				activation = new ReLU();
			} else {
				printf("(%s) unknown activation\n", name.c_str());
				exit(1);
			}
			argc -= 2; argv += 2;
		} else { //if (arg == "-h") {
			printf("Argument usage: \n");
			printf("  -b <nb_batch>  -s <seq_len> -l <layer_size> -a <activation> -w <weight_initialization>\n");
			printf("  Activations: \"tanh\"|\"sigmoid\"|\"iden\"|\"relu\"\n");
		}
	}

	arma_rng::set_seed_random(); // REMOVE LATER

	Model* m  = new Model(); // argument is input_dim of model
	m->setBatchSize(nb_batch);
	m->setSeqLen(seq_len);
	//m->setLayerSize(layer_size);
	//m->setIsRecurrent(is_recurrent);
	m->setInitializationType(initialization_type);

	testRecurrentModelBias1(m, layer_size, is_recurrent, activation);
}
