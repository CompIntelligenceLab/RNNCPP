#include <unordered_map>

class Func { 
public:
	REAL alpha;
	Func(REAL alpha) { this->alpha = alpha; }
	virtual REAL operator()(REAL x) = 0;
};
class ExpFunc : public Func { 
public:
	ExpFunc(REAL alpha) : Func(alpha) {;}
	REAL operator()(REAL x) { return exp(-alpha*x); }
};


void updateWeightsSumConstraint(Model* m, Layer* d1, Layer* d2, Layer* e1, Layer* e2)
{
	// Sum of weights (d1,d2) + (e1,e2) = constant
	// deal with weight constraints (weights are frozon, but update by hand)
	// in this case, sum of two weights is unity
	// w1 + w2 = 1 ==> delta(w1) + delta(w2) = 0. 
	// call w1 = w and w2 = 1-w. Compute delta(w)
	// Compute  delta(w) = delta(w1) - delta(w2)
	// w1 -= lr * delta(w)
	// w2 += lr * delta(w)

	WEIGHT& deltaw1 = m->getConnection(d1, d2)->getDelta();
	//deltaw1.print("deltaw1");
	WEIGHT& deltaw2 = m->getConnection(e1, e2)->getDelta();
	//deltaw2.print("deltaw2");
	WEIGHT delta = deltaw1 - deltaw2;
	WEIGHT& w1 = m->getConnection(d1, d2)->getWeight();
	WEIGHT& w2 = m->getConnection(e1, e2)->getWeight();
	REAL lr = m->getLearningRate();
	//delta.print("delta");
	w1 -= 0.001 * lr * delta;
	w2 += 0.001 * lr * delta;
	//printf("w1, w2= %f, %f\n", w1[0,0], w2[0,0]);

	m->getConnection(d1, d2)->weight_history.push_back(w1);
	m->getConnection(e1, e2)->weight_history.push_back(w2);
}
//----------------------------------------------------------------------
void updateWeightsSumConstraint(Model* m, Layer* d1, Layer* d2, Layer* e1, Layer* e2, Layer* f1, Layer* f2)
{
	// Let w1, w2 and w3 = 1 - w1 - w2
	// dL/dw1 = -dL/dw3 
	// dL/dw2 = -dL/dw3 
	// dL/dw = 
	// delta3 = -delta1 - delta2 = 0
	WEIGHT& dd1 = m->getConnection(d1, d2)->getDelta();
	WEIGHT& dd2 = m->getConnection(e1, e2)->getDelta();
	WEIGHT& dd3 = m->getConnection(f1, f2)->getDelta();
	WEIGHT& w1 = m->getConnection(d1, d2)->getWeight();
	WEIGHT& w2 = m->getConnection(e1, e2)->getWeight();
	WEIGHT& w3 = m->getConnection(f1, f2)->getWeight();
	// weights are w1, w2, 1-w1-w2
	WEIGHT delta13 = dd1 - dd3;
	WEIGHT delta23 = dd2 - dd3;
	REAL lr = m->getLearningRate();
	w1 -= .001 * lr * delta13;
	w2 -= .001 * lr * delta23;
	w3 -= .001 * lr * (-delta13-delta23);
}
//----------------------------------------------------------------------
int getCharData(Model* m, 
        std::vector<VF2D_F>& net_inputs, 
		std::vector<VF2D_F>& net_exact, 
    	std::string& input_data,
		std::unordered_map<char, int>& c_int,
		std::vector<char>& int_c,
		arma::field<VI>& hot)
{
	int nb_chars = input_data.size();
	printf("nb_characters = %d\n", nb_chars);
	printf("seq_len= %d\n", m->getSeqLen());
	printf("nb_batch= %d\n", m->getBatchSize());
	printf("input_dim= %d\n", m->getInputDim());
	int nb_batch = m->getBatchSize();
	int input_dim = m->getInputDim();
	int seq_len = m->getSeqLen();

	VF2D_F vf2d;
	//printf("nb_batch= %d\n", nb_batch); exit(0);
	U::createMat(vf2d, nb_batch, input_dim, seq_len);

	VF2D_F vf2d_exact;
	U::createMat(vf2d_exact, nb_batch, input_dim, seq_len);

	int nb_samples = (input_data.size() / (input_dim * nb_batch * seq_len));
	printf("nb_samples= %d\n", nb_samples);

	#if 1
	for (int i=0; i < 65; i++) {
		printf("c_int.at: %d\n", c_int.at(input_data[i]));
	}

	// vf2d[batch](nb_chars, seq_len)

	// Assume nb batches is 1 for now. Write code and debug it. 

	// We will write a function to extract the next batch

	printf("input_data.size= %d\n", input_data.size());
	for (int i=0; i < nb_samples-1; i++) {
  	for (int b=0; b < nb_batch; b++) {
		int base = b * seq_len * input_dim; //  ideally, need loop on input_dim
			for (int s=0; s < seq_len; s++) {
				//printf("b,i,j= %d, %d, %d\n", b, i, j);
				//printf("1, base+j+seq_len*i = %d\n", base+s+seq_len*i);
				// NEED A LOOP OVER INPUTS  (one-hot)
				printf("base= %d\n", base+s+seq_len*i);
				printf("input_data: %c\n", input_data[base+s+seq_len*i]);
				vf2d[b](0, s)       = c_int.at(input_data[base + s + seq_len*i]);
				printf("vf2d: %d\n", vf2d[b](0,s));
				vf2d_exact[b](0, s) = c_int.at(input_data[base + s + seq_len*i + 1]); // wasteful of memory
				exit(0);
			}
		}
		net_inputs.push_back(vf2d);
		net_exact.push_back(vf2d_exact);
	}

	for (int s=0; s < 10; s++) {
		printf("input, exact: %d, %d\n", vf2d[0](0, s), vf2d_exact[0](0, s));
	}
	exit(0);
	#endif
	return nb_samples;
}
//----------------------------------------------------------------------
int getData(Model* m, std::vector<VF2D_F>& net_inputs, std::vector<VF2D_F>& net_exact, VF1D& x, VF1D& ytarget)
{
	//------------------------------------------------------------------
	// SOLVE ODE  dy/dt = -alpha y
	// Determine alpha, given a curve YT (y target) = exp(-2 t)
	// Initial condition on alpha: alpha(0) = 1.0
	// I expect the neural net to return alpha=2 when the loss function is minimized. 

	int seq_len = m->getSeqLen();
	int nb_batch = m->getBatchSize();
	int input_dim  = 1;

	int npts = 600;
	printf("npts= %d\n", npts); 
	printf("seq_len= %d\n", seq_len); 

	// npts should be a multiple of seq_len
	npts = (npts / seq_len) * seq_len; 
	// npts should be a multiple of nb_batch
	npts = (npts / nb_batch) * nb_batch;
	x.resize(npts);
	ytarget.resize(npts);

	//VF1D ytarget(npts);
	//VF1D x(npts);   // abscissa
	REAL delx;
	delx = 0.005; // orig
	//delx = .001;  // will this work for uneven time steps? dt = .1, but there is a multiplier: alpha in front of it. 
	                 // Some form of normalization will probably be necessary to scale out effect of dt (discrete time step)
	m->dt = delx;
	REAL alpha_target = 2.;
	REAL alpha_initial = 1.;  // should evolve towards alpha_target

	// this works (duplicates Mark Lambert's case)
	//REAL alpha_target  = 1.;
	//REAL alpha_initial = 2.;  // should evolve towards alpha_target

	Func& fun1 = *(new ExpFunc(alpha_target));
	Func& fun2 = *(new ExpFunc(-.1));


	// Choose the function to use to determine differential equation

	for (int i=0; i < npts; i++) {
		x[i] = i*delx;
		//ytarget[i] = fun1(x[i]);
		// scheme has problems with this function. I cannot find an equation that works. 
		// I tried 3 nodes in parallel. Did not work. 
		ytarget[i] = 0.5 * (fun1(x[i]) + fun2(x[i]));
		//ytarget[i] = fun2(x[i]);
	}

	// set all alphas to alpha_initial
	LAYERS layers = m->getLayers();

	for (int l=0; l < layers.size(); l++) {
		Layer* layer = layers[l];
		//printf("l= %d\n", l);
		// layers without parameters will ignore this call
		layer->getActivation().setParam(0, alpha_initial); // 1st parameter
		layer->getActivation().setDt(m->dt);
	}

	// Assume nb_batch=1 for now. Extract a sequence of seq_len elements to input
	// input into prediction followed by backprop followed by parameter updates.
	// What I want is a data structure: 
	// VF2D_F[nb_batch][nb_inputs, seq_len] = VF2D_F[1][1, seq_len]

	int nb_samples = npts / seq_len; 
	//std::vector<VF2D_F> net_inputs, net_exact;
	VF2D_F vf2d;
	//printf("nb_batch= %d\n", nb_batch); exit(0);
	U::createMat(vf2d, nb_batch, 1, seq_len);

	VF2D_F vf2d_exact;
	U::createMat(vf2d_exact, nb_batch, 1, seq_len);

	#if 0
	// Assumes nb_batch = 1 and input dimensionality = 1
	for (int i=0; i < nb_samples-1; i++) {
		for (int j=0; j < seq_len; j++) {
			vf2d[0](0, j)       = ytarget(j + seq_len*i);
			vf2d_exact[0](0, j) = ytarget(j + seq_len*i + 1);
		}
		net_inputs.push_back(vf2d);
		net_exact.push_back(vf2d_exact);
	}
	#endif

	// Assumes nb_batch >= 0, and input dimensionality = 1
	//printf("nb_batch= %d\n",nb_batch); exit(0);

	// nb_samples must be a multiple of the batch size
	nb_samples = npts / (nb_batch * seq_len); 
	U::print(vf2d_exact, "vf2d_exact");
	U::print(vf2d, "vf2d");
	U::print(ytarget, "ytarget");
	//exit(0);

printf("nb_samples= %d\n",nb_samples);
printf("seq_len= %d\n",seq_len);
printf("nb_batch= ", nb_batch);


for (int i=0; i < nb_samples-1; i++) {
  for (int b=0; b < nb_batch; b++) {
	int base = b * seq_len * input_dim; //  ideally, need loop on input_dim
		for (int j=0; j < seq_len; j++) {
			//printf("b,i,j= %d, %d, %d\n", b, i, j);
			//printf("1, base+j+seq_len*i = %d\n", base+j+seq_len*i);
			vf2d[b](0, j)       = ytarget(base + j + seq_len*i);
			vf2d_exact[b](0, j) = ytarget(base + j + seq_len*i + 1); // wasteful of memory
		}
	}
	net_inputs.push_back(vf2d);
	net_exact.push_back(vf2d_exact);
}
//exit(0);



	delete &fun1;
	delete &fun2;
	return nb_samples;
}
//----------------------------------------------------------------------
#ifdef DEBUG
void printDerivativeBreakdown(Model* m) 
{
	printf("\n\n****************************************************\n");
	printf("printDerivativeBreakdown: DELTAS of LAYERS for all times: \n");
	int seq_len = m->getSeqLen();
	LAYERS layers = m->getLayers();
	CONNECTIONS connections = m->getConnections();

	for (int l=0; l < layers.size(); l++) {
		layers[l]->printSummary();
		for (int s=0; s < seq_len; s++) {
			printf("--- s= %d\n", s);
			//printf("layers[s]->deltas[%d] = %ld\n", s, layers.deltas[s]);
			layers[l]->deltas[s].print("delta layer-");
		}
	}

	connections.push_back(layers[1]->getConnection());
	layers[1]->getConnection()->printSummary();

	printf("\n\n****************************************************\n");
	printf("printDerivativeBreakdown: DELTAS of CONNECTIONS for all times: \n");
	for (int c=0; c < connections.size(); c++) {
		Connection* con = connections[c];
		if (con->deltas.size()) {
			con->printSummary();
		}
		for (int s=0; s < con->deltas.size(); s++) {
			printf("--- s= %d\n", s);
			con->deltas[s].print("delta con");
		}
	}
}
#endif
//----------------------------------------------------------------------
WEIGHT weightDerivative(Model* m, Connection& con, REAL fd_inc, VF2D_F& xf, VF2D_F& exact)
{
	WEIGHT w0 = con.getWeight();
	int rrows = w0.n_rows;
	int ccols = w0.n_cols;
	dLdw = arma::Mat<REAL>(size(w0));
	dLdw.zeros();
	Objective* mse = new MeanSquareError();

	for (int rr=0; rr < rrows; rr++) {
	for (int cc=0; cc < ccols; cc++)
	{
		WEIGHT& wp = con.getWeight(); 
		wp(rr,cc) += fd_inc;
		wp.print("wp");
		VF2D_F pred_n = m->predictViaConnectionsBias(xf);
		pred_n[0].raw_print(cout, "pred_n");
		//printf("inside weightDerivative\n"); exit(0);

		WEIGHT& wm = con.getWeight(); 
		wm(rr,cc) -= (2.*fd_inc);
		VF2D_F pred_p = m->predictViaConnectionsBias(xf);
		wp(rr,cc) += fd_inc; // I had forgotten this. 

		// Sum the loss over the sequences
		LOSS loss_p = (*mse)(exact, pred_p); // LOSS is a row of REALs
		LOSS loss_n = (*mse)(exact, pred_n);

		// take the derivative of batch 0, of the loss (summed over the sequences)
		dLdw(rr, cc) = (arma::sum(loss_n(0)) - arma::sum(loss_p(0))) / (2.*fd_inc);
		for (int b=1; b < loss_p.n_rows; b++) {
			dLdw(rr, cc) += (arma::sum(loss_n(b)) - arma::sum(loss_p(b))) / (2.*fd_inc);
		}
	}}

	delete mse;
	return dLdw;
}
//----------------------------------------------------------------------
BIAS biasFDDerivative(Model* m, Layer& layer, REAL fd_inc, VF2D_F& xf, VF2D_F& exact)
{
	BIAS bias = layer.getBias();
	int layer_size = layer.getLayerSize();
	dLdb = BIAS(size(bias));
	dLdb.zeros();
	Objective* mse = new MeanSquareError();

	printf("=== biasFDDerivative ===\n");

	for (int rr=0; rr < layer_size; rr++)
	{
		BIAS& biasp = layer.getBias();
		biasp(rr) += fd_inc;
		VF2D_F pred_n = m->predictViaConnectionsBias(xf);
		pred_n[0].raw_print(cout, "pred_n");

		BIAS& biasm = layer.getBias(); 
		biasm(rr) -= (2.*fd_inc);
		VF2D_F pred_p = m->predictViaConnectionsBias(xf);
		pred_p[0].raw_print(cout, "pred_p");
		biasm(rr) += fd_inc;

		// Sum the loss over the sequences
		LOSS loss_p = (*mse)(exact, pred_p); // LOSS is a row of REALs
		LOSS loss_n = (*mse)(exact, pred_n);

		// take the derivative of batch 0, of the loss (summed over the sequences)
		dLdb(rr) = (arma::sum(loss_n(0)) - arma::sum(loss_p(0))) / (2.*fd_inc);
		for (int b=1; b < loss_p.n_rows; b++) {
			dLdb(rr) += (arma::sum(loss_n(b)) - arma::sum(loss_p(b))) / (2.*fd_inc);
		}
	}
	delete mse;
	return dLdb;
}
//----------------------------------------------------------------------
VF1D activationParamsFDDerivative(Model* m, Layer& layer, REAL fd_inc, VF2D_F& xf, VF2D_F& exact)
{
	//BIAS bias = layer.getBias();
	Activation& activation = layer.getActivation();
	VF1D activation_delta = layer.getActivationDelta();

	int layer_size = layer.getLayerSize();
	VF1D dLdp = VF1D(activation.getNbParams());   // dL/d(Parameters)
	dLdp.zeros();
	Objective* mse = new MeanSquareError();
	printf("layer name: %s\n", layer.getName().c_str());
	const VF1D& params = activation.getParams();
	//for (int i=0; i < 10; i++) { printf("x params[%d]= %f\n", i, params[i]); }
	//exit(0);
	//ppp[0] = 3.;
	//ppp.resize(10);
	/*
	printf("params.size()= %d\n", params.size());
	printf("params.n_rows= %d\n", params.n_rows);
	printf("params.n_cols= %d\n", params.n_cols);
	printf("params: %f\n", params[0,0]);
	printf("params: %f\n", params[0]);
	printf("params: %f\n", params[1]);
	U::print(params, "params"); // ERROR Why does not work? 
	params.print("params");
	*/

	printf("************ activationParamsFDDerivative ****************\n");
	for (int rr=0; rr < activation.getNbParams(); rr++)
	{
		if (activation.isFrozen(rr)) {
			continue; 
		}

		REAL param = params[rr];
		printf("param= %f\n", param);
		param += fd_inc;
		activation.setParam(rr, param);
		printf("param+inc= %f\n", param);
		VF2D_F pred_n = m->predictViaConnectionsBias(xf);
		pred_n.print("pred_n");

		param -= (2.*fd_inc);
		activation.setParam(rr, param);
		printf("param-inc= %f\n", param);
		VF2D_F pred_p = m->predictViaConnectionsBias(xf);
		pred_p.print("pred_p");
		param += fd_inc;
		activation.setParam(rr, param);
		printf("return to original value, param= %f\n", params[rr]);
		activation.setParam(rr, param);

		// Sum the loss over the sequences
		LOSS loss_p = (*mse)(exact, pred_p); // LOSS is a row of REALs
		LOSS loss_n = (*mse)(exact, pred_n);

		loss_p.print("loss_p");
		loss_n.print("loss_n");

		// take the derivative of batch 0, of the loss (summed over the sequences)
		dLdp(rr) = (arma::sum(loss_n(0)) - arma::sum(loss_p(0))) / (2.*fd_inc);
		for (int b=1; b < loss_p.n_rows; b++) {
			dLdp(rr) += (arma::sum(loss_n(b)) - arma::sum(loss_p(b))) / (2.*fd_inc);
		}
		dLdp.print("dLdp");
	}
	printf("activationParamsFDDerivative\n"); //exit(0);
	delete mse;
	return dLdp;
}
//----------------------------------------------------------------------
std::vector<WEIGHT> runTest(Model* m, REAL inc, VF2D_F& xf, VF2D_F& exact)
{
	VF2D_F pred;

	CONNECTIONS connections = m->getConnections();
	const LAYERS& layers = m->getLayers();
	const VF2D_F& inputs = layers[1]->getInputs();
	const VF2D_F& outputs = layers[0]->getOutputs();

	// How to compute the less function
	pred = m->predictViaConnectionsBias(xf);
	//pred[0].raw_print(std::cout, "pred");
	//exit(0);

	Objective* obj = m->getObjective();
	const LOSS& loss = (*obj)(exact, pred);

	printf("********************** ENTER BACKPROP **************************\n");
	m->backPropagationViaConnectionsRecursion(exact, pred); // Add sequence effect. 
	printf("********************** EXIT BACKPROP **************************\n");

	std::vector<Connection*> conn;
	std::vector<BIAS> bias_fd, bias_bp;
	std::vector<WEIGHT> weight_fd, weight_bp;
	std::vector<VF1D> param_fd, param_bp;
	std::vector<REAL> w_norm, w_abs_err_norm, w_rel_err_norm;
	std::vector<REAL> b_norm, b_abs_err_norm, b_rel_err_norm;

	// NON-RECURRENT WEIGHTS and CONNECTIONS
	for (int c=0; c < connections.size(); c++) {
		Connection* con = connections[c];
		if (con->from == 0) continue;
		conn.push_back(con);
		WEIGHT weight_fd_ = weightDerivative(m, *con, inc, xf, exact);
		weight_fd.push_back(weight_fd_);
	 	WEIGHT weight_bp_ = con->getDelta();
		weight_bp.push_back(weight_bp_);
		//weight_bp_.print("weight_bp_");
		//printf("norm(weight_bp_)= %f\n", arma::norm(weight_bp_));
		w_norm.push_back(arma::norm(weight_bp_));
		//connections[c]->printSummary();
	}
	//printf("after non-recurrent FD weights\n"); exit(0);

	// BIASES
	for (int l=0; l < layers.size(); l++) {
		if (layers[l]->type == "input") continue;
		BIAS bias_fd_ = biasFDDerivative(m, *layers[l], inc, xf, exact);
		//bias_fd_.print("bias_fd_");
		bias_fd.push_back(bias_fd_);
    	BIAS bias_bp_ = layers[l]->getBiasDelta();
		//bias_bp_.print("bias_bp_");
		bias_bp.push_back(bias_bp_);
		b_norm.push_back(arma::norm(bias_bp_));
	}

	// RECURRENT WEIGHTS and CONNECTIONS
	for (int l=0; l < layers.size(); l++) {
		Connection* con = layers[l]->getConnection();
		// when seq_len=1, recurrence has no effect. 
		if (con && layers[l]->getSeqLen() > 1) { 
			//con->printSummary("con"); 
			conn.push_back(con);
			//con->getWeight().print("*weight*");
		 	WEIGHT weight_fd_ = weightDerivative(m, *con, inc, xf, exact);
			weight_fd.push_back(weight_fd_);
		 	WEIGHT weight_bp_ = con->getDelta();
		 	//weight_bp_.print("..............weight_bp_"); 
			weight_bp.push_back(weight_bp_);
			//weight_bp_.print("weight_bp_");
			//printf("norm(weight_bp_)= %f\n", arma::norm(weight_bp_));
			w_norm.push_back(arma::norm(weight_bp_));
		}
	}

	// ACTIVATION PARAMETERS
	for (int l=0; l < layers.size(); l++) {
		Layer* layer = layers[l];
		if (layers[l]->type == "input") continue;
		Activation& activation = layer->getActivation();
		int nbParams = activation.getNbParams();

		VF1D param_fd_ = activationParamsFDDerivative(m, *layer, inc, xf, exact);
		param_fd.push_back(param_fd_);
		VF1D param_bp_ = layer->getActivationDelta();
		param_bp.push_back(param_bp_);
	}

	// compute L2 norms of various quantities

	printf("---------------------------------------------------\n");
	printf("Relative ERRORS for weight derivatives: \n");

	printf("\n***********************************************************\n");
	printf("\n      FINITE-Difference vs BACKPROP ***********************\n");
	printf("        List of Connections                                  \n");

	for (int i=0; i < weight_fd.size(); i++) {
		WEIGHT abs_err = (weight_fd[i] - weight_bp[i]);
		WEIGHT rel_err = abs_err / weight_bp[i];
		abs_err = arma::abs(abs_err);
		rel_err = arma::abs(rel_err);
		REAL abs_err_norm = arma::norm(abs_err);
		REAL rel_err_norm = arma::norm(rel_err);
		int imx = rel_err.index_max();
		REAL wgt_imx = weight_bp[i](imx);
		REAL rel_err_imx = rel_err(imx);
		//REAL err_norm_inf = arma::norm(err, "inf");

		printf("\nConnection %d\n", i); conn[i]->printSummary();
		conn[i]->getWeight().print("*** Connection weight matrix ***");
		weight_bp[i].raw_print(cout, "bp derivatives: ");
		weight_fd[i].raw_print(cout, "fd derivatives: ");
		printf("    norms: w,abs_err,rel_err= %14.7e, %14.7e, %14.7e\n", w_norm[i], abs_err_norm, rel_err_norm);
		//printf("    max rel error: %14.7e at index weight_bp: %f\n", rel_err_imx, wgt_imx);

		printf("\n\n");
		//printf("weight: w,abs,rel= %f, %f, %f, norm_inf= %f\n", w_norm[i], abs_err_norm, err_norm, err_norm_inf);

	#if 0
	if (i == 1) { // recurrent weight
		int nr = weight_fd[1].n_rows;
		int nc = weight_fd[1].n_cols;
		nr = (nr > 3) ? 3 : nr;
		nc = (nc > 3) ? 3 : nc;
		VF2D& ww = weight_bp[i];

		#if 0
		U::print(ww, "w_11");
		for (int r=0; r < nr; r++) {
		for (int c=0; c < nc; c++) {
			printf("<<<weight_bp/weight_fd(%d,%d)= %14.7f, %14.7f\n", r, c, weight_bp[i](r,c), weight_fd[i](r,c));  
		}}
		#endif
	}
	#endif

		#if 0
		printf("----------\n");
		printf("...d1-d1: ");  weight_bp[i].raw_print(cout, "weight_bp");
		printf("   d1-d1: ");  abs_err.print("weight abs err");
		printf("   d1-d1: ");  rel_err.print("weight rel err");
		#endif
	}

	#if 1
	printf("Relative ERRORS for bias derivatives: \n");

	for (int i=0; i < bias_fd.size(); i++) {
		BIAS abs_err = (bias_fd[i] - bias_bp[i]);
		bias_fd[i].raw_print(cout, "bias_fd");
		bias_bp[i].raw_print(cout, "bias_bp");
		BIAS err = (bias_fd[i] - bias_bp[i]) / bias_bp[i];
		REAL abs_err_norm = arma::norm(abs_err);
		REAL err_norm     = arma::norm(err);
		printf("\n"); printf("bias: norms: bias,abs,rel= %f, %f, %f\n", b_norm[i], abs_err_norm, err_norm);
		#if 0
		printf("   d1-d1: "); bias_bp[i].print("bias bp");
		printf("   d1-d1: "); abs_err.print("bias abs err");
		printf("   d1-d1: "); err.print("bias rel error");
		#endif
	}
	#endif

	printf("\n\nRelative ERRORS for activation parameter derivatives: \n");

	printf("param_fd.size()= %d\n", param_fd.size());
	printf("***********************\n"); //exit(0);

	for (int i=0; i < param_fd.size(); i++) {
		printf("********** layer %d ***********\n");
		VF1D abs_err = (param_fd[i] - param_bp[i]);
		VF1D err     = (param_fd[i] - param_bp[i]) / param_bp[i];
		REAL abs_err_norm = arma::norm(abs_err); 
		REAL     err_norm = arma::norm(    err); 
		param_fd[i].print("fd derivatives of activation parameters");
		param_bp[i].print("bp derivatives of activation parameters");
		printf("\n"); printf("Activation params: norms: abs,rel= %f, %f\n", abs_err_norm, err_norm);
	}

	// return vector of weights
	std::vector<WEIGHT> ws;
	ws.push_back(weight_bp[1]); // recurrent weight
	//printf("Shapes of weight_bp\n");
	//U::print(weight_bp[0], "bp[0]");
	//U::print(weight_bp[1], "bp[1]");
	return ws;
}
//----------------------------------------------------------------------
void predictAndBackProp(Model* m, VF2D_F& xf, VF2D_F& exact)
{
	VF2D_F pred = m->predictViaConnectionsBias(xf);
	m->backPropagationViaConnectionsRecursion(exact, pred); // Add sequence effect. 
}
//----------------------------------------------------------------------
Model* processArguments(int argc, char** argv)
{
// arguments: -b nb_batch, -l layer_size, -s seq_len, -s is_recursive

    int nb_epochs = 10;
    int nb_batch = 1;
    int layer_size = 1;
    int seq_len = 1;
    int is_recurrent = 1;
	int nb_serial_layers = 1; // do not count input layer
	int nb_parallel_layers = 1; // do not count input layer
	std::string obj_err_type = "abs";
	REAL learning_rate = 1.e-2; 

	REAL inc;
	string activation_type;
	//Activation* activation = new Identity(); 
	std::string initialization_type;
	initialization_type = "xavier";

	argv++; 
	argc--; 

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
		} else if (arg == "-lr") {
			learning_rate = atof(argv[1]);
			argc -= 2; argv += 2;
		} else if (arg == "-i") {
			inc = atof(argv[1]);
			argc -= 2; argv += 2;
		} else if (arg == "-e") {
			nb_epochs = atof(argv[1]);
			argc -= 2; argv += 2;
		} else if (arg == "-w") {
			initialization_type = argv[1];
			argc -= 2; argv += 2;
			printf("init type: %s\n", initialization_type.c_str());
			// "xavier", "xavier_iden", "unity", "gaussian", 
		} else if (arg == "-nsl") { // number serial layers
			nb_serial_layers = atoi(argv[1]);
			argc -= 2; argv += 2;
		} else if (arg == "-npl") {  // number parallel layers
			nb_parallel_layers = atoi(argv[1]);
			argc -= 2; argv += 2;
		} else if (arg == "-l") {
			layer_size = atoi(argv[1]);
			argc -= 2; argv += 2;
		} else if (arg == "-oe") {  // rel/abs error_type for objective least squares function
			obj_err_type = argv[1];
			argc -= 2; argv += 2;
		} else if (arg == "-a") {
			std::string name = argv[1];
			activation_type = name;

			// NEED an ACTIVATION FACTORY

			argc -= 2; argv += 2;
		} else { //if (arg == "-h") 
			printf("Argument (%s) not found. \nArgument usage: \n", arg.c_str());
			printf("  -b <nb_batch>  -s <seq_len> -nb <nb_layers> -l <layer_size> -a <activation> -w <weight_initialization>\n");
			printf("  Activations: \"tanh\"|\"sigmoid\"|\"iden\"|\"relu\"\n");
			exit(0);
		}
	}

	//arma_rng::set_seed_random(); // REMOVE LATER
	arma_rng::set_seed(100); // REMOVE LATER

	Model* m  = new Model(); // argument is input_dim of model
	m->setBatchSize(nb_batch);
	m->setSeqLen(seq_len);
	m->setInitializationType(initialization_type);

	m->layer_size = layer_size;
	m->is_recurrent = is_recurrent;
	m->inc = inc;
	m->nb_serial_layers =   nb_serial_layers;
	m->nb_parallel_layers = nb_parallel_layers;
	m->nb_epochs = nb_epochs;
	m->setLearningRate(learning_rate); // default lr
	m->obj_err_type = obj_err_type;

	for (int j=0; j < nb_parallel_layers; j++) {
	for (int i=0; i < nb_serial_layers; i++) {
		if (activation_type == "tanh") {
			m->activations.push_back(new Tanh());
		} else if (activation_type == "iden") {
			m->activations.push_back(new Identity());
		} else if (activation_type == "sigmoid") {
			m->activations.push_back(new Sigmoid());
		} else if (activation_type == "relu") {
			m->activations.push_back(new ReLU());
		} else if (activation_type == "decayde") {
			m->activations.push_back(new DecayDE());
		} else {
			printf("(%s) unknown activation\n", activation_type.c_str());
			exit(1);
		}
	}}

	//delete activation;
	return m;
}
