
WEIGHT weightDerivative(Model* m, Connection& con, float inc, VF2D_F& xf, VF2D_F& exact)
{
	// I'd expect the code to work with nb_batch=1 
	//printf("********** ENTER weightDerivative *************, \n");

	WEIGHT w0 = con.getWeight();
	int rrows = w0.n_rows;
	int ccols = w0.n_cols;
	dLdw = arma::Mat<float>(size(w0));
	dLdw.zeros();
	Objective* mse = new MeanSquareError();

	for (int rr=0; rr < rrows; rr++) {
	for (int cc=0; cc < ccols; cc++) {

		WEIGHT& wp = con.getWeight(); 
		wp(rr,cc) += inc;
		VF2D_F pred_n = m->predictViaConnectionsBias(xf);

		WEIGHT& wm = con.getWeight(); 
		wm(rr,cc) -= (2.*inc);
		VF2D_F pred_p = m->predictViaConnectionsBias(xf);

		// Sum the loss over the sequences
		LOSS loss_p = (*mse)(exact, pred_p); // LOSS is a row of floats
		//U::print(pred_p, "pred_p"); 
		//U::print(loss_p, "loss_p"); 
		//loss_p.print("loss_p");
		//exit(0);
		LOSS loss_n = (*mse)(exact, pred_n);

		//loss_n(0) = arma::sum(loss_n(0), 1);
		//loss_p(0) = arma::sum(loss_p(0), 1);
		//U::print(loss_n, "loss_n");
		//loss_n.print("loss_n");
		//loss_n(0).print("loss_n(0)");
		//exit(0);

		// take the derivative of batch 0, of the loss (summed over the sequences)
		dLdw(rr, cc) = (arma::sum(loss_n(0)) - arma::sum(loss_p(0))) / (2.*inc);
	}}
	//con.printSummary("weightDerivative");
	//dLdw.print("dLdw");
	//printf("********** EXIT weightDerivative *************, \n");
	return dLdw;
}
//----------------------------------------------------------------------
BIAS biasDerivative(Model* m, Layer& layer, float inc, VF2D_F& xf, VF2D_F& exact)
{
	// I'd expect the code to work with nb_batch=1 
	//printf("********** ENTER biasDerivative *************, \n");

	BIAS bias = layer.getBias();
	int layer_size = layer.getLayerSize();
	dLdb = BIAS(size(bias));
	dLdb.zeros();
	Objective* mse = new MeanSquareError();

	for (int rr=0; rr < layer_size; rr++) {

		BIAS& biasp = layer.getBias();
		biasp(rr) += inc;
		VF2D_F pred_n = m->predictViaConnectionsBias(xf);

		BIAS& biasm = layer.getBias(); 
		biasm(rr) -= (2.*inc);
		VF2D_F pred_p = m->predictViaConnectionsBias(xf);

		// Sum the loss over the sequences
		LOSS loss_p = (*mse)(exact, pred_p); // LOSS is a row of floats
		LOSS loss_n = (*mse)(exact, pred_n);

		// take the derivative of batch 0, of the loss (summed over the sequences)
		dLdb(rr) = (arma::sum(loss_n(0)) - arma::sum(loss_p(0))) / (2.*inc);
	}
	//printf("********** EXIT biasDerivative *************, \n");
	return dLdb;
}
//----------------------------------------------------------------------
void runTest(Model* m, float inc, VF2D_F& xf, VF2D_F& exact)
{
	VF2D_F pred;

	CONNECTIONS connections = m->getConnections();
	const LAYERS& layers = m->getLayers();
	const VF2D_F& inputs = layers[1]->getInputs();
	const VF2D_F& outputs = layers[0]->getOutputs();

	// How to compute the less function
	pred = m->predictViaConnectionsBias(xf);


	#if 0
	layers[0]->getInputs().print("layer 0 inputs");
	layers[1]->getInputs().print("layer 1 inputs");
	layers[1]->getInputs()(0).print("layer 1 inputs(0)");
	layers[1]->getInputs()(0).col(0).print("layer 1 inputs(0).col(0)");
	inputs[0].col(0).print("in.col(0)");
	#endif

	printf("......\n");
	layers[0]->getOutputs().print("layer 0 outputs");
	layers[1]->getOutputs().print("layer 1 outputs");
	outputs[0].col(0).print("layer 0, out.col(0)");
	outputs[0].col(1).print("layer 0, out.col(1)");
	inputs[0].col(0).print("layer 1, in.col(0)");
	inputs[0].col(1).print("layer 1, in.col(1)");
	printf("XXXXXXXXXXXXXX\n"); 

	Objective* obj = m->getObjective();
	const LOSS& loss = (*obj)(exact, pred);

	printf("before back\n");
	m->backPropagationViaConnectionsRecursion(exact, pred); // Add sequence effect. 

	#if 0
	layers[0]->getInputs().print("layer 0 inputs");
	layers[1]->getInputs().print("layer 1 inputs");
	layers[1]->getInputs()(0).print("layer 1 inputs(0)");
	layers[1]->getInputs()(0).col(0).print("layer 1 inputs(0).col(0)");
	inputs[0].col(0).print("in.col(0)");
	#endif

	printf("......\n");
	layers[0]->getOutputs().print("layer 0 outputs");
	layers[1]->getOutputs().print("layer 1 outputs");
	outputs[0].col(0).print("layer 0, out.col(0)");
	outputs[0].col(1).print("layer 0, out.col(1)");
	inputs[0].col(0).print("layer 1, in.col(0)");
	inputs[0].col(1).print("layer 1, in.col(1)");
	printf("XXXXXXXXXXXXXX\n"); 


	printf("YYYYYYYYYYYYYYYYYYYYYYYYYYYY\n");
	const VF2D_F& inputs1 = layers[1]->getInputs();
	const VF2D_F& outputs1 = layers[1]->getOutputs();
	inputs1[0].col(0).print("(x) in.col(0)");
	outputs1[0].col(0).print("(y) out.col(0)");
	//inputs1[0].col(1).print("in.col(1)");
	//outputs1[0].col(1).print("out.col(1)");

	layers[1]->getActivation().jacobian(inputs1[0].col(0), outputs[0].col(0)).print("jacobian");
	connections[0]->printSummary(); connections[0]->getDelta().print();
	connections[1]->printSummary(); connections[1]->getDelta().print();
	//printf("XXXXXXXXXXXXXX\n"); exit(0);
	//exit(0);

	std::vector<BIAS> bias_fd, bias_bp;
	std::vector<WEIGHT> weight_fd, weight_bp;

	for (int c=0; c < connections.size(); c++) {
		Connection* con = connections[c];
		if (con->from == 0) continue;
		WEIGHT weight_fd_ = weightDerivative(m, *con, inc, xf, exact);
		weight_fd.push_back(weight_fd_);
	 	WEIGHT weight_bp_ = con->getDelta();
		weight_bp.push_back(weight_bp_);
		connections[c]->printSummary();
	}

	for (int l=0; l < layers.size(); l++) {
		if (layers[l]->type == "input") continue;
		BIAS bias_fd_ = biasDerivative(m, *layers[l], inc, xf, exact);
		bias_fd_.print("bias_fd_");
		bias_fd.push_back(bias_fd_);
    	BIAS bias_bp_ = layers[l]->getBiasDelta();
		bias_bp_.print("bias_bp_");
		bias_bp.push_back(bias_bp_);
		//layers[l]->printSummary();

		Connection* con = layers[l]->getConnection();
		if (con) {
		 	WEIGHT weight_fd_ = weightDerivative(m, *con, inc, xf, exact);
			weight_fd.push_back(weight_fd_);
		 	WEIGHT weight_bp_ = con->getDelta();
			weight_bp.push_back(weight_bp_);
			//layers[l]->getConnection()->printSummary();
		}
	}

	printf("Relative ERRORS for weight derivatives for batch 0: \n");

	for (int i=0; i < weight_fd.size(); i++) {
		WEIGHT err = (weight_fd[i] - weight_bp[i]) / weight_bp[i];
		printf("   wght: ");  weight_bp[i].print("weight_bp");
		printf("   d1-d1: "); err.print("weight rel err");
	}

	printf("Relative ERRORS for bias derivatives for batch 0: \n");

	for (int i=0; i < bias_fd.size(); i++) {
		BIAS err = (bias_fd[i] - bias_bp[i]) / bias_bp[i];
		printf("   d1-d1: "); err.print();
	}
}
