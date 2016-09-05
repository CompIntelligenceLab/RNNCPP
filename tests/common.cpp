
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

	// How to compute the less function
	pred = m->predictViaConnectionsBias(xf);

	Objective* obj = m->getObjective();
	const LOSS& loss = (*obj)(exact, pred);

	m->backPropagationViaConnectionsRecursion(exact, pred); // Add sequence effect. 
	
	WEIGHT& delta_bp_1 = connections[1]->getDelta();
	Layer* d1 = layers[1];
	Connection* c1 = d1->getConnection();
	WEIGHT& delta_bp_3 = c1->getDelta();

	WEIGHT delta_fd_1 = weightDerivative(m, *connections[1], inc, xf, exact);

	if (!d1) {
		printf("common.cpp: d1 should be nonzero!\n");
		exit(1);
	}

	//std::vector<BIAS> bias_fd;
	//std::vector<BIAS> bias_bp;
	//std::vector<WEIGHT> weight_fd;
	//std::vector<WEIGHT> weight_bp;

	WEIGHT delta_fd_3 = weightDerivative(m, *d1->getConnection(), inc, xf, exact);

	WEIGHT weight_err_1 = (delta_fd_1 - delta_bp_1) / delta_bp_1;
	WEIGHT weight_err_3 = (delta_fd_3 - delta_bp_3) / delta_bp_3;

	BIAS bias_fd_1 = biasDerivative(m, *layers[1], inc, xf, exact);
    BIAS bias_bp_1 = layers[1]->getBiasDelta();
	BIAS bias_err_1 = (bias_fd_1 - bias_bp_1) / bias_bp_1;


	printf("Relative ERRORS for weight derivatives for batch 0: \n");
	printf("input-d1: "); weight_err_1.print();
	printf("   d1-d1: "); weight_err_3.print();

	printf("\nRelative ERRORS for bias derivatives for batch 0: \n");
	printf("layer d1: "); bias_err_1.print();

}
