
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
		VF2D_F pred_n = m->predictViaConnectionsBias(xf);

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

	for (int rr=0; rr < layer_size; rr++)
	{
		BIAS& biasp = layer.getBias();
		biasp(rr) += fd_inc;
		VF2D_F pred_n = m->predictViaConnectionsBias(xf);

		BIAS& biasm = layer.getBias(); 
		biasm(rr) -= (2.*fd_inc);
		VF2D_F pred_p = m->predictViaConnectionsBias(xf);

		// Sum the loss over the sequences
		LOSS loss_p = (*mse)(exact, pred_p); // LOSS is a row of REALs
		LOSS loss_n = (*mse)(exact, pred_n);

		// take the derivative of batch 0, of the loss (summed over the sequences)
		dLdb(rr) = (arma::sum(loss_n(0)) - arma::sum(loss_p(0))) / (2.*fd_inc);
		for (int b=1; b < loss_p.n_rows; b++) {
			dLdb(rr) += (arma::sum(loss_n(b)) - arma::sum(loss_p(b))) / (2.*fd_inc);
		}
	}
	return dLdb;
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
		weight_bp[i].print("bp derivatives: ");
		weight_fd[i].print("fd derivatives: ");
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
		BIAS err = (bias_fd[i] - bias_bp[i]) / bias_bp[i];
		REAL abs_err_norm = arma::norm(abs_err);
		REAL err_norm     = arma::norm(err);
		printf("\n"); printf("bias: norms: w,abs,rel= %f, %f, %f\n", b_norm[i], abs_err_norm, err_norm);
		#if 0
		printf("   d1-d1: "); bias_bp[i].print("bias bp");
		printf("   d1-d1: "); abs_err.print("bias abs err");
		printf("   d1-d1: "); err.print("bias rel error");
		#endif
	}
	#endif

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
			} else if (name == "decayde") {
				activation = new DecayDE();
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

	arma_rng::set_seed_random(); // REMOVE LATER
	//arma_rng::set_seed(100); // REMOVE LATER

	Model* m  = new Model(); // argument is input_dim of model
	m->setBatchSize(nb_batch);
	m->setSeqLen(seq_len);
	m->setInitializationType(initialization_type);

	m->layer_size = layer_size;
	m->is_recurrent = is_recurrent;
	m->activation = activation;
	m->inc = inc;

	return m;

	//testRecurrentModelBias1(m, layer_size, is_recurrent, activation, inc);
}
