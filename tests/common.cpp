
WEIGHT weightDerivative(Model* m, Connection& con, float inc, VF2D_F& xf, VF2D_F& exact)
{
	WEIGHT w0 = con.getWeight();
	int rrows = w0.n_rows;
	int ccols = w0.n_cols;
	dLdw = arma::Mat<float>(size(w0));
	dLdw.zeros();
	Objective* mse = new MeanSquareError();

	for (int rr=0; rr < rrows; rr++) {
	for (int cc=0; cc < ccols; cc++)
	{
		WEIGHT& wp = con.getWeight(); 
		wp(rr,cc) += inc;
		VF2D_F pred_n = m->predictViaConnectionsBias(xf);

		WEIGHT& wm = con.getWeight(); 
		wm(rr,cc) -= (2.*inc);
		VF2D_F pred_p = m->predictViaConnectionsBias(xf);

		// Sum the loss over the sequences
		LOSS loss_p = (*mse)(exact, pred_p); // LOSS is a row of floats
		LOSS loss_n = (*mse)(exact, pred_n);

		// take the derivative of batch 0, of the loss (summed over the sequences)
		dLdw(rr, cc) = (arma::sum(loss_n(0)) - arma::sum(loss_p(0))) / (2.*inc);
		for (int b=1; b < loss_p.n_rows; b++) {
			dLdw(rr, cc) += (arma::sum(loss_n(b)) - arma::sum(loss_p(b))) / (2.*inc);
		}
	}}
	return dLdw;
}
//----------------------------------------------------------------------
BIAS biasDerivative(Model* m, Layer& layer, float inc, VF2D_F& xf, VF2D_F& exact)
{
	BIAS bias = layer.getBias();
	int layer_size = layer.getLayerSize();
	dLdb = BIAS(size(bias));
	dLdb.zeros();
	Objective* mse = new MeanSquareError();

	for (int rr=0; rr < layer_size; rr++)
	{
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
		for (int b=1; b < loss_p.n_rows; b++) {
			dLdb(rr) += (arma::sum(loss_n(b)) - arma::sum(loss_p(b))) / (2.*inc);
		}
	}
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
	//pred.print("pred");
	//exit(0);

	Objective* obj = m->getObjective();
	const LOSS& loss = (*obj)(exact, pred);
	//loss.print("loss");

	m->backPropagationViaConnectionsRecursion(exact, pred); // Add sequence effect. 

	std::vector<BIAS> bias_fd, bias_bp;
	std::vector<WEIGHT> weight_fd, weight_bp;
	std::vector<Connection*> conn;
	std::vector<float> w_norm, w_abs_err_norm, w_rel_err_norm;
	std::vector<float> b_norm, b_abs_err_norm, b_rel_err_norm;

	for (int c=0; c < connections.size(); c++) {
		Connection* con = connections[c];
		if (con->from == 0) continue;
		conn.push_back(con);
		WEIGHT weight_fd_ = weightDerivative(m, *con, inc, xf, exact);
		weight_fd.push_back(weight_fd_);
	 	WEIGHT weight_bp_ = con->getDelta();
		weight_bp.push_back(weight_bp_);
		w_norm.push_back(arma::norm(weight_bp_));
		//connections[c]->printSummary();
	}

	for (int l=0; l < layers.size(); l++) {
		if (layers[l]->type == "input") continue;
		BIAS bias_fd_ = biasDerivative(m, *layers[l], inc, xf, exact);
		//bias_fd_.print("bias_fd_");
		bias_fd.push_back(bias_fd_);
    	BIAS bias_bp_ = layers[l]->getBiasDelta();
		//bias_bp_.print("bias_bp_");
		bias_bp.push_back(bias_bp_);
		b_norm.push_back(arma::norm(bias_bp_));

		Connection* con = layers[l]->getConnection();
		// when seq_len=1, recurrence has no effect. 
		if (con and layers[l]->getSeqLen() > 1) {
			con->printSummary("con"); 
			conn.push_back(con);
			//con->getWeight().print("*weight*");
		 	WEIGHT weight_fd_ = weightDerivative(m, *con, inc, xf, exact);
			weight_fd.push_back(weight_fd_);
		 	WEIGHT weight_bp_ = con->getDelta();
			weight_bp.push_back(weight_bp_);
			w_norm.push_back(arma::norm(weight_bp_));
		}
	}

	// compute L2 norms of various quantities

	printf("++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
	printf("Relative ERRORS for weight derivatives: \n");

	for (int i=0; i < weight_fd.size(); i++) {
		printf("\n"), conn[i]->printSummary();
		WEIGHT abs_err = (weight_fd[i] - weight_bp[i]);
		WEIGHT rel_err = abs_err / weight_bp[i];
		abs_err = arma::abs(abs_err);
		rel_err = arma::abs(rel_err);
		float abs_err_norm = arma::norm(abs_err);
		float rel_err_norm = arma::norm(rel_err);
		int imx = rel_err.index_max();
		float wgt_imx = weight_bp[i](imx);
		float rel_err_imx = rel_err(imx);
		//float err_norm_inf = arma::norm(err, "inf");

		conn[i]->getWeight().print("*** weight ***");
		printf("weight: w,abs,rel= %f, %f, %f, norm_inf= %f\n", w_norm[i], abs_err_norm, rel_err_norm);
		printf("max rel error: %f at weight_bp: %f\n", rel_err_imx, wgt_imx);
		//printf("weight: w,abs,rel= %f, %f, %f, norm_inf= %f\n", w_norm[i], abs_err_norm, err_norm, err_norm_inf);
		#if 0
		printf("   d1-d1: ");  weight_bp[i].print("weight_bp");
		printf("   d1-d1: ");  abs_err.print("weight abs err");
		printf("   d1-d1: ");  rel_err.print("weight rel err");
		#endif
	}

	printf("Relative ERRORS for bias derivatives: \n");

	for (int i=0; i < bias_fd.size(); i++) {
		BIAS abs_err = (bias_fd[i] - bias_bp[i]);
		BIAS err = (bias_fd[i] - bias_bp[i]) / bias_bp[i];
		float abs_err_norm = arma::norm(abs_err);
		float err_norm     = arma::norm(err);
		printf("\n"); printf("bias: w,abs,rel= %f, %f, %f\n", b_norm[i], abs_err_norm, err_norm);
		#if 0
		printf("   d1-d1: "); bias_bp[i].print("bias bp");
		printf("   d1-d1: "); abs_err.print("bias abs err");
		printf("   d1-d1: "); err.print("bias rel error");
		#endif
	}
}
//----------------------------------------------------------------------
void predictAndBackProp(Model* m, VF2D_F& xf, VF2D_F& exact)
{
	VF2D_F pred = m->predictViaConnectionsBias(xf);
	m->backPropagationViaConnectionsRecursion(exact, pred); // Add sequence effect. 
}
//----------------------------------------------------------------------
