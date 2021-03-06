#ifndef __OBJECTIVE_H__
#define __OBJECTIVE_H__

#include <string>
#include "typedefs.h"

class Objective
{
protected:
	REAL learning_rate;
	std::string name;
	LOSS loss;  // One loss per batch and per sequence (use field for consistency)
	VF2D_F gradient; // One gradient with respect to argument 
	static int counter;

public:
	Objective(std::string name="objective");
	virtual ~Objective();
	Objective(const Objective&);
	const Objective& operator=(const Objective&);
	virtual void print(std::string name= "");
	virtual void setName(std::string name) { this->name = name; }
	virtual const std::string getName() const { return name; }
	virtual void setLoss(LOSS loss) { this->loss = loss; }
	virtual const LOSS& getLoss() const { return loss; }
	virtual VF2D_F& getGradient() { return gradient; }
	
	virtual void computeLoss(const VF2D_F& exact, const VF2D_F& predict) = 0;
	virtual void computeGradient(const VF2D_F& exact, const VF2D_F& predict) = 0;

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

#endif
