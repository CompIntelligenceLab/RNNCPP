#ifndef __OBJECTIVE_H__
#define __OBJECTIVE_H__

#include <string>
#include "typedefs.h"
#include "globals.h"

class Objective
{
protected:
	// equal to average of exact over sequence and dimensions
	VF1D weight; // factor by which to multiply the objective function
	REAL learning_rate;
	std::string name;
	LOSS loss;  // One loss per batch and per sequence (use field for consistency)
	VF2D_F gradient; // One gradient with respect to argument 
	static int counter;
	std::string error_type; // absolute or relative error for least mean square
	Globals* globals;  // disadvantage: same value in all objective functions if more than one
	                  // perhaps not, since globals is not shared between objectives and member variables are public.

public:
	Objective(std::string name="objective");
	virtual void setGlobals(Globals* globals);
	virtual ~Objective();
	Objective(const Objective&);
	const Objective& operator=(const Objective&);
	virtual void print(std::string name= "");
	virtual void setName(std::string name) { this->name = name; }
	virtual const std::string getName() const { return name; }
	virtual void setLoss(LOSS loss) { this->loss = loss; }
	virtual const LOSS& getLoss() const { return loss; }
	virtual VF2D_F& getGradient() { return gradient; }
	virtual void setWeight(VF1D weight) { this->weight = weight; }
	virtual VF1D& getWeight() { return this->weight; }
	virtual void setErrorType(std::string error_type) { this->error_type = error_type; }
	virtual std::string getErrorType() { return(error_type); }
	
	virtual void computeLoss(const VF2D_F& exact, const VF2D_F& predict) = 0;
	virtual void computeGradient(const VF2D_F& exact, const VF2D_F& predict) = 0;
	virtual arma::Row<REAL> computeLossOneBatch(const VF2D& exact, const VF2D& predict) {;}
	virtual VF2D computeGradientOneBatch(const VF2D& exact, const VF2D& predict) {;}

	virtual const LOSS& operator()(const VF2D_F& exact, const VF2D_F& predict) {
		computeLoss(exact, predict);
		return getLoss();
	}
};
//----------------------------------------------------------------------
class MeanSquareError : public Objective
{
private:
public:
	MeanSquareError(std::string name="mse");
	~MeanSquareError();
	MeanSquareError(const MeanSquareError&);

	// Use default assignment (works fine because there are no pointers among member variables)
	//const MeanSquareError& MeanSquareError=(const MeanSquareError&);

	/** sum_{batches} (predict - exact)^2 */
	void computeLoss(const VF2D_F& exact, const VF2D_F& predict);
	void computeGradient(const VF2D_F& exact, const VF2D_F& predict);
};
//----------------------------------------------------------------------
class WeightedMeanSquareError : public Objective
{
private:
public:
	WeightedMeanSquareError(std::string name="wmse");
	~WeightedMeanSquareError();
	WeightedMeanSquareError(const WeightedMeanSquareError&);

	// Use default assignment (works fine because there are no pointers among member variables)
	//const MeanSquareError& MeanSquareError=(const MeanSquareError&);

	/** sum_{batches} (predict - exact)^2 */
	void computeLoss(const VF2D_F& exact, const VF2D_F& predict);
	void computeGradient(const VF2D_F& exact, const VF2D_F& predict);
};
//----------------------------------------------------------------------
class LogMeanSquareError : public Objective
{
// Logarithm of mean square error
private:
public:
	LogMeanSquareError(std::string name="logmse");
	~LogMeanSquareError();
	LogMeanSquareError(const LogMeanSquareError&);

	// Use default assignment (works fine because there are no pointers among member variables)
	//const MeanSquareError& MeanSquareError=(const MeanSquareError&);

	/** sum_{batches} (predict - exact)^2 */
	void computeLoss(const VF2D_F& exact, const VF2D_F& predict);
	void computeGradient(const VF2D_F& exact, const VF2D_F& predict);
};
//----------------------------------------------------------------------
class BinaryCrossEntropy : public Objective
{
private:
public:
	BinaryCrossEntropy(std::string name="mse");
	~BinaryCrossEntropy();
	BinaryCrossEntropy(const BinaryCrossEntropy&);

	// Use default assignment (works fine because there are no pointers among member variables)
	//const BinaryCrossEntropy& BinaryCrossEntropy=(const BinaryCrossEntropy&);

	/** sum_{batches} (predict - exact)^2 */
	void computeLoss(const VF2D_F& exact, const VF2D_F& predict);
	void computeGradient(const VF2D_F& exact, const VF2D_F& predict);
};
//----------------------------------------------------------------------
class CrossEntropy : public Objective
{
private:
public:
	CrossEntropy(std::string name="mse");
	~CrossEntropy();
	CrossEntropy(const CrossEntropy&);

	// Use default assignment (works fine because there are no pointers among member variables)
	//const CrossEntropy& CrossEntropy=(const CrossEntropy&);

	/** sum_{batches} (predict - exact)^2 */
	void computeLoss(const VF2D_F& exact, const VF2D_F& predict);
	void computeGradient(const VF2D_F& exact, const VF2D_F& predict);
};
//----------------------------------------------------------------------
class GMM1D : public Objective
{
private:
	REAL reg;

public:
	GMM1D(std::string name="gmm1d");
	~GMM1D();
	GMM1D(const GMM1D&);

	// Use default assignment (works fine because there are no pointers among member variables)
	//const GMM1D& GMM1D=(const GMM1D&);

	/** sum_{batches} (predict - exact)^2 */
	void computeLoss(const VF2D_F& exact, const VF2D_F& predict);
	void computeGradient(const VF2D_F& exact, const VF2D_F& predict);
	arma::Row<REAL> computeLossOneBatch(const VF2D& exact, const VF2D& predict);
	VF2D computeGradientOneBatch(const VF2D& exact, const VF2D& predict);
};
//----------------------------------------------------------------------

#endif

