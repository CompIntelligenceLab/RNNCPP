#include <math.h>
#include "../common.h"
#include <string>
#include <cstdlib>

REAL derivLoss(int k, VF1D& z0, VF1D& e, VF2D& w11, REAL alpha, int m0, int n0, std::vector<VF2D>& ws);
REAL predict(int k, VF1D& z0, VF1D& e, VF2D& w11, REAL alpha, int m0, int n0, std::vector<VF2D>& ws);

void diagRecurrenceTest(Model* m, Layer* input, Layer* d1, VF2D_F& xf, VF2D_F& exact)
{
	REAL x0       = .45;
	REAL ex0      = .75; // exact value
	REAL alpha    = .98;  // Diagonal element for weight matrix

	int nb_batch = m->getBatchSize();
	//printf("nb_batch= %d\n", nb_batch);
	int input_dim = m->getInputLayers()[0]->getInputDim();
	//printf("input_dim= %d\n", input_dim);
	int output_dim = m->getOutputLayers()[0]->getOutputDim();
	int seq_len = m->getSeqLen();

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
		conn->computeWeightTranspose();
		ww01 = w_01; // TEMPORARY
	}
	#endif
		WEIGHT w_11;
		VF2D total_deriv;
		Connection* conn;
		conn = d1->getConnection();
		if (conn) {
			conn->printSummary(); 
			w_11 = conn->getWeight();
			total_deriv = VF2D(size(w_11));
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
				//printf("power = %d\n", s);
			}

			// Compute exact derivatives for one weight element, for starters. 
			REAL alpha = w_11(0,0);
			int nr = w_11.n_rows;
			int nc = w_11.n_cols;
			REAL alphap = pow(alpha, seq_len);

			int k;
			
			U::print(xf, "xf");
			U::print(ww01, "ww01");
			xf.print("xf");
			ww01.print("ww01");

			VF1D z0 = ww01*xf(0).col(0); // ERROR

			nr = total_deriv.n_rows;
			nc = total_deriv.n_cols;
			//nr = nr > 3 ? 3 : nr;
			//nc = nc > 3 ? 3 : nc;

			//*********************************************************
			//         EXACT DERIVATIVE        ************************
			//*********************************************************
			// derivative of L(k) with respect to w_11(m0, n0)

			//std::vector<REAL> 

			z0.print("z0");
			w_11.print("w11");
			exact.print("exact");

			for (int m0=0; m0 < nr; m0++) {
			for (int n0=0; n0 < nc; n0++) {
				//if (n0 != 0 || m0 != 0) continue; // ONLY HANDLE ONE DERIVATIVE FOR NOW
				printf("\n--w11--- m0,n0= %d, %d\n", m0, n0);
				printf("--> derivLoss\n");
				total_deriv(m0,n0) = 0.;
				for (int k=0; k < seq_len; k++) {
					VF1D e = exact(0).col(k);  // exact can be arbitrary
					printf("\n*****************************************************\n");
					printf("*******      DERIVLOSS     *************************\n");
					REAL dl = derivLoss(k, z0, e, w_11, alpha, m0, n0, ws);
					total_deriv(m0,n0) += dl;
					printf("--- dl = %14.7f\n", dl);
				}
				VF1D e = exact(0).col(seq_len-1);  // exact can be arbitrary
				predict(seq_len-1, z0, e, w_11, alpha, m0, n0, ws);
			}}
			//exit(0);
		}

		std::vector<WEIGHT> wss;
		REAL inc = 0.01;

		/***********
		***********/

		wss = runTest(m, inc, xf, exact); 
		total_deriv.print("exact deriv (total_deriv)");
		wss[0].print("ws[0]");
		WEIGHT abs_err = wss[0] - total_deriv;
		WEIGHT rel_err = abs_err / wss[0];
		wss[0].print("wss[0]");
		U::print(abs_err, "abs_err");
		U::print(rel_err, "rel_err");
		REAL abs_err_norm = arma::norm(abs_err);
		REAL rel_err_norm = arma::norm(rel_err);
        int imx = rel_err.index_max();
	    REAL wgt_imx = wss[0](imx);
	    REAL abs_err_max = abs_err(imx);
	    REAL rel_err_max = rel_err(imx);
		printf("abs_err_norm: %14.7e\n", abs_err_norm);
		printf("rel_err_norm: %14.7e\n", rel_err_norm);
		printf("max abs error: %14.7e\n", abs_err_max);
		printf("max rel error: %14.7e\n", rel_err_max);
		printf("  at weight_bp: %14.7f\n",  wgt_imx);
		exit(0);



		printf("\n*****************************************************************\n");
		LAYERS layers = m->getLayers();
		CONNECTIONS connections = m->getConnections();

		layers[1]->getConnection()->printSummary("layers[1]->getConnection");
		layers[1]->getConnection()->getDelta().print("layers[1]->getConnection, delta");

		{
			WEIGHT abs_err = wss[0] - total_deriv;
			WEIGHT rel_err = abs_err % wss[0];
			total_deriv.print("total_deriv");
			//rel_err.print("rel_err btwn back_prop and analytic");
			REAL rel_err_norm = arma::norm(rel_err);
			printf("--> rel_err_norm(w11 vs analytic)  = %f\n", rel_err_norm);
		}

		printf("\n\n*** Derivative computed via backprop via RNN model\n");
		printDerivativeBreakdown(m);
		exit(0);
}
//----------------------------------------------------------------------
void testRecurrentModelBias1(Model* m, int layer_size, int is_recurrent, Activation* activation, REAL inc) 
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


	int output_dim = m->getOutputLayers()[0]->getOutputDim();

	VF2D_F xf(nb_batch), exact(nb_batch);
		//xf.print("xf"); 
	//U::print(xf, ".xf"); exit(0);

	diagRecurrenceTest(m, input, d1, xf, exact);
	exit(0);

	#if 1
	{
		BIAS& bias_1 = d1->getBias();
		bias_1(0) = 0.; //bias1;
	}
	#endif

	//================================
	std::vector<WEIGHT> wss;
	wss = runTest(m, inc, xf, exact); 
	exit(0);

	predictAndBackProp(m, xf, exact);
	exit(0);
}
//----------------------------------------------------------------------
REAL predict(int k, VF1D& z0, VF1D& e, VF2D& w11, REAL alpha, int m0, int n0, std::vector<VF2D>& ws)
{
	VF1D x = z0;
	printf("iteration: 0, x= %14.7f\n", x(0));  //x.print("x: ");

	for (int i=0; i < k; i++) {
		x = w11*x;
		printf("iteration: %d, x= %14.7f\n", i+1, x(0));  //x.print("x: ");
	}
}
//----------------------------------------------------------------------
REAL derivLoss(int k, VF1D& z0, VF1D& e, VF2D& w11, REAL alpha, int m0, int n0, std::vector<VF2D>& ws)
{
	VF2D x(1,1);
	#if 0
	printf("INSIDE derivloss\n");
	printf("alpha= %f\n", alpha);
	printf("m0, n0= %d, %d\n", m0, n0);
	printf("w11= %f\n", w11(0,0));
	printf("e= %f\n", e(0));
	printf("k= %d\n", k);
	printf("\n");
	printf("z0= %f\n", z0(0));
	z0.print("z0");
	#endif

	REAL alphak = pow(alpha,k);
	VF1D l = 2.*(alphak*z0-e); //.t(); // correct
	std::cout.precision(11);
	//std::cout.setf(ios::fixed);
	l.raw_print("\n derivLoss, d(loss)/da), l.. (loss)");

	REAL Lprime = l[m0] * k * pow(alpha, k-1) * z0[n0];
	//printf("2*(w11*z0-e0)= %f\n",  2.*(w11(0,0)*z0(0)-e(0)));


	//REAL dLdw11 = 2.*(w11(0,0)*z0(0)-e(0)) * z0(0);
	//printf("*****************************\n");
	//printf("analytical dLdw11= %f\n", dLdw11);
	//printf("derivLoss dLdw11= %f\n", Lprime);
	//printf("*****************************\n");
	return Lprime;
}
	//	REAL dLdw11 = 2.*(w11*w01*x0 - e1) * w01*x0;
//----------------------------------------------------------------------
int main(int argc, char* argv[])
{
// arguments: -b nb_batch, -l layer_size, -s seq_len, -s is_recursive

    int nb_batch = 1;
    int layer_size = 1;
    int seq_len = 1;
    int is_recurrent = 1;
	REAL inc;
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
		} else if (arg == "-i") {
			inc = atof(argv[1]);
			argc -= 2; argv += 2;
		} else if (arg == "-w") { // weight initialization
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
		} else { //if (arg == "-h") 
			printf("Argument usage: \n");
			printf("  -b <nb_batch>  -s <seq_len> -l <layer_size> -a <activation> -w <weight_initialization>\n");
			printf("  Activations: \"tanh\"|\"sigmoid\"|\"iden\"|\"relu\"\n");
		}
	}

	//arma_rng::set_seed_random(); // REMOVE LATER
	arma_rng::set_seed(100); // REMOVE LATER

	Model* m  = new Model(); // argument is input_dim of model
	m->setBatchSize(nb_batch);
	m->setSeqLen(seq_len);
	m->setInitializationType(initialization_type);

	testRecurrentModelBias1(m, layer_size, is_recurrent, activation, inc);
}
