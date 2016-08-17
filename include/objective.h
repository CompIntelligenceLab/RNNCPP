#ifndef __OBJECTIVE_H__
#define __OBJECTIVE_H__

#include <string>
#include "typedefs.h"

class Objective
{
protected:
	float learning_rate;
	std::string name;
	VF1D_F loss;  // One loss per batch and per sequence (use field for consistency)
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
	virtual void setLoss(VF1D_F loss) { this->loss = loss; }
	virtual VF1D_F& getLoss() { return loss; }
	virtual VF2D_F& getGradient() { return gradient; }
	
	virtual void computeLoss(const VF2D_F& exact, const VF2D_F& predict) = 0;
	virtual void computeGradient(const VF2D_F& exact, const VF2D_F& predict) = 0;

	virtual VF1D_F& operator()(const VF2D_F& exact, const VF2D_F& predict) {
		computeLoss(exact, predict);
		return getLoss();
	}
};

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

#endif
