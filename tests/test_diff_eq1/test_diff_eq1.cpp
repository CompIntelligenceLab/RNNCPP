#include <math.h>
#include "../common.h"
#include <string>
#include <cstdlib>

REAL derivLoss(int k, VF1D& z0, VF1D& e, VF2D& w11, REAL alpha, int m0, int n0, std::vector<VF2D>& ws);
REAL predict(int k, VF1D& z0, VF1D& e, VF2D& w11, REAL alpha, int m0, int n0, std::vector<VF2D>& ws);

void diagRecurrenceTest(Model* m, Layer* input, Layer* d1, VF2D_F& xf, VF2D_F& exact, REAL inc)
{
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
		int seq_len = m->getSeqLen();
		WEIGHT w_11;
		VF2D total_deriv;
		Connection* conn;
		conn = d1->getConnection();
		if (conn) {
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
			//w.print("w power");
			//ws[0].print("ws[0]");
			//ws[1].print("ws[1]");
			//ws[seq_len-1].print("ws[seq_len-1]");
			//ws[seq_len].print("ws[seq_len]");
			//exit(0);

			// Compute exact derivatives for one weight element, for starters. 
			REAL alpha = w_11(0,0);
			//xf.print("xf"); 
			//printf("alpha= %f\n", alpha); 
			int nr = w_11.n_rows;
			int nc = w_11.n_cols;
			REAL alphap = pow(alpha, seq_len);
			//printf("seq_len= %d\n", seq_len);
			//printf("alphap= %f\n", alphap);
			//exact(0).print("exact(0)");

			int k;
			//U::print(ww01, "ww01");
			//xf(0).col(0).print("xf(0).col(0)");
			//xf(0).col(1).print("xf(0).col(1)");
			//z0.print("z0");
			//VF1D y = xf;
			//e.print("e");
			
			VF1D z0 = ww01*xf(0).col(0); // ERROR
			#if 0
			printf("*********************  z0 ******************\n");
			printf("ww01= %f\n", ww01(0,0));
			printf("xf(0).col(0)= %f\n", xf(0).col(0)(0));
			printf("z0= %f\n", z0(0));
			printf("*****************************************\n");
			#endif

			nr = total_deriv.n_rows;
			nc = total_deriv.n_cols;
			//nr = nr > 3 ? 3 : nr;
			//nc = nc > 3 ? 3 : nc;

			// derivative of L(k) with respect to w_11(m0, n0)
			for (int m0=0; m0 < nr; m0++) {
			for (int n0=0; n0 < nc; n0++) {
				printf("\n--w11--- m0,n0= %d, %d\n", m0, n0);
				//printf("seq_len= %d\n", seq_len);
				total_deriv(m0,n0) = 0.;
				for (int k=0; k < seq_len; k++) {
					VF1D e = exact(0).col(k);  // exact can be arbitrary
					REAL dl = derivLoss(k, z0, e, w_11, alpha, m0, n0, ws);
					total_deriv(m0,n0) += dl;
					printf("--- dl = %14.7f\n", dl);
				}
				VF1D e = exact(0).col(seq_len-1);  // exact can be arbitrary
				predict(seq_len-1, z0, e, w_11, alpha, m0, n0, ws);
				//printf("total_deriv(%d,%d)= %14.7f\n", m0, n0, total_deriv(m0,n0));
			}}
			//exit(0);

			#if 0
			// KEEP FOR TESTING
			printf("\n\n Direction computation of derivative of L wrt w_{11} when s=2 and layer_size=1\n");
			{
				Connection* conn = m->getConnection(input, d1);
				WEIGHT& w_01 = conn->getWeight();
				REAL w11 = w_11(0,0);
				REAL w01 = w_01(0,0);
				REAL e0 = exact(0).col(0)(0);
				REAL e1 = exact(0).col(1)(0);
				REAL x0 = xf(0)(0,0);  // x1 = 0 for this solution
				REAL dLdw11 = 2.*(w11*w01*x0 - e1) * w01*x0;
				REAL dLdw01 = 2.*(w11*w01*x0-e1)*w11*x0 + 2.*(w01*x0-e0) * x0;
				printf("dLdw01= %14.7f\n", dLdw01);
				printf("dLdw11= %14.7f\n", dLdw11);
			}
			#endif
		}

		std::vector<WEIGHT> wss;
		//REAL inc = 0.001;

		/***********
		***********/

		xf.print("xf");
		exact.print("exact");


		// missing xf and exact
		//wss = runTest(m, inc, xf, exact); 
		printf("\n<<< Exit after runTest >>>\n");
		exit(0);

		printf("\n*****************************************************************\n");
		LAYERS layers = m->getLayers();
		CONNECTIONS connections = m->getConnections();

		printf("input-recurrent\n"); connections[0]->printSummary();
		connections[0]->getDelta().print("delta");
		printf("recurrent\n"); connections[1]->printSummary();
		connections[1]->getDelta().print("delta");

		layers[0]->printSummary();
		layers[0]->getDelta().print("delta");
		layers[1]->printSummary();
		layers[1]->getDelta().print("delta");

		layers[1]->getConnection()->printSummary("layers[1]->getConnection");
		layers[1]->getConnection()->getDelta().print("layers[1]->getConnection, delta");

		{
			U::print(wss[0], "wss");
			U::print(total_deriv, "total_deriv");
			WEIGHT abs_err = wss[0] - total_deriv;
			WEIGHT rel_err = abs_err % wss[0];
			wss[0].print("wss[0]");
			total_deriv.print("total_deriv");
			//rel_err.print("rel_err btwn back_prop and analytic");
			REAL rel_err_norm = arma::norm(rel_err);
			printf("--> rel_err_norm(w11 vs analytic)  = %f\n", rel_err_norm);
		}
}
//----------------------------------------------------------------------
//void testRecurrentModelBias1(Model* m, int layer_size, int is_recurrent, Activation* activation, REAL inc) 
void testDiffEq1(Model* m)
{
	//testRecurrentModelBias1(m, layer_size, is_recurrent, activation, inc);
	int layer_size = m->layer_size;
	int is_recurrent = m->is_recurrent;
	Activation* activation = m->activation;
	REAL inc = m->inc;


	printf("\n\n\n");
	printf("=============== BEGIN test_recurrent_model_bias2  =======================\n");

	//================================
	int seq_len = m->getSeqLen();
	int nb_batch = m->getBatchSize();
	int input_dim  = 1;

	// Layers automatically adjust ther input_dim to match the output_dim of the previous layer
	// 2 is the dimensionality of the data
	// the names have a counter value attached to it, so there is no duplication. 
	Layer* input = new InputLayer(input_dim, "input_layer");
	Layer *d1;

	is_recurrent = 0;

	if (is_recurrent) {
		d1    = new RecurrentLayer(layer_size, "rdense");
	} else {
		d1    = new DenseLayer(layer_size, "rdense");
	}

	m->add(0,     input);
	m->add(input, d1);

	// input should always be identity activation
	input->setActivation(new Identity()); 
       d1->setActivation(activation); 

	m->addInputLayer(input);
	m->addOutputLayer(d1);

	m->printSummary();
	m->connectionOrderClean(); // no print statements

	m->initializeWeights();

	// Initialize xf and exact
	VF2D_F xf, exact;
	int input_size = input->getLayerSize();
	U::createMat(xf, nb_batch, input_size, seq_len);
	U::createMat(exact, nb_batch, layer_size, seq_len);
	U::print(xf, "xf"); //exit(0);
	U::print(exact, "exact"); //exit(0);
	//exit(0);

	for (int b=0; b < xf.n_rows; b++) {
		xf[b] = arma::randu<VF2D>(input_size, seq_len); //size(xf[b]));
		exact[b] = arma::randu<VF2D>(layer_size, seq_len); //size(xf[b]));
	}
	xf.print("xf"); 
	exact.print("exact"); 
	U::print(xf, "xf"); //exit(0);
	U::print(exact, "exact"); //exit(0);
	//exit(0);

	// SOME KIND OF MATRIX INCOMPATIBILITY. That is because exact has the wrong dimensions. 
	runTest(m, inc, xf, exact);
	printf("gordon\n");
	exit(0);

	//===========================================

	REAL w01 = .4;
	REAL w11 = .6;
	REAL bias1 = 0.; //-.7;  // single layer of size 1 ==> single bias
	//w01       = 1.;
	//w11       = 1.;
	//w22       = 1.;
	REAL x0       = .45;
	REAL x1       = .75;
	REAL ex0      = .75; // exact value
	REAL ex1      = .85; // exact value
	VF2D a(layer_size+1, seq_len); // assume al dimensions = 1
	VF2D z(layer_size+1, seq_len); // assume al dimensions = 1

	int output_dim = m->getOutputLayers()[0]->getOutputDim();

	//VF2D_F xf(nb_batch), exact(nb_batch);
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


	// Set to 1 to run the test with diagonal recurrent matrix
	diagRecurrenceTest(m, input, d1, xf, exact, inc);
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
	printf("z0= %f\n", z0(0));
	printf("k= %d\n", k);
	printf("\n");
	#endif
	REAL alphak = pow(alpha,k);
	VF1D l = 2.*(alphak*z0-e); //.t();
	std::cout.precision(11);
	//std::cout.setf(ios::fixed);
	l.raw_print("\n derivLoss, l.. (loss)");

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

	Model* m = processArguments(argc, argv);
	testDiffEq1(m);
}

