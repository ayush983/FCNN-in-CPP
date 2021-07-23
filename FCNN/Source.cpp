#include<vector>
#include<iostream>
#include<cstdlib>
#include<fstream>
#include<string>
#include<sstream>
#include<cassert>
#include<chrono>
using namespace std;

class TrainingData
{
public:
	TrainingData(const string filename);
	bool isEof(void) { return m_trainingDataFile.eof(); }
	void getTopology(vector<unsigned>& topology);

	unsigned getNextInputs(vector<double>& inputVals);
	void getTargetOutputs(vector<double>& targetOutputVals);

private:
	ifstream m_trainingDataFile;
};

void TrainingData::getTopology(vector<unsigned>& topology)
{
	string line;
	string label;

	getline(m_trainingDataFile, line);
	stringstream ss(line);
	ss >> label;
	cout << label;
	if (this->isEof() || label.compare("topology:") != 0) {
		abort();
	}

	while (!ss.eof()) {
		unsigned  n;
		ss >> n;
		topology.push_back(n);
	}
	return;
}

TrainingData::TrainingData(const string filename)
{
	m_trainingDataFile.open(filename.c_str());
}

unsigned TrainingData::getNextInputs(vector<double>& inputVals)
{
	inputVals.clear();
	string line;
	getline(m_trainingDataFile, line);
	stringstream ss(line);


	double val;
	ss >> val;
	int count = 0;
	while (int(val) != -1)
	{
		inputVals.push_back(val);
		getline(m_trainingDataFile, line);
		stringstream ss(line);
		ss >> val;
		count++;
	}


	return inputVals.size();
}
void TrainingData::getTargetOutputs(vector<double>& targetOutputVals)
{
	targetOutputVals.clear();

	string line;
	getline(m_trainingDataFile, line);
	stringstream ss(line);
	double val;
	ss >> val;
	//targetOutputVals.push_back(val);
	int i = 0;
	while (i < 10)
	{
		if (i == int(val)) targetOutputVals.push_back(1);
		else targetOutputVals.push_back(0.0);
		++i;
	}
}
struct Connection
{
	double weight;
	double deltaWeight;
};
class Neuron;
typedef vector<Neuron> Layer;
class Neuron {
public:
	Neuron(unsigned numOutputs, unsigned myIndex);
	void setOutputVal(double val) { m_outputVal = val; }
	double getOutputVal(void) const { return m_outputVal; }
	void feedForward(const Layer& prevLayer);
	void calcOutputGradients(double targetVal, int softm);
	void calcHiddenGradients(const Layer& nextLayer);
	void updateInputWeights(Layer& prevLayer);
	double calcWeightedSum(Layer& prevLayer);
	void ffsoftm(const Layer& prevLayer, double constant);


private:
	static double eta;
	static double alpha;
	static double sigmoid(double x);
	static double softmax(double x, double constant);
	static double softmaxDerivative(double x);
	static double sigmoidDerivative(double x);
	static double randomWeight(void) { return rand() / double(RAND_MAX); }
	double m_outputVal;
	double sumDOW(const Layer& nextlayer)const;
	vector<Connection> m_outputWeights;
	unsigned m_myIndex;
	double m_gradient;


};
double Neuron::eta = 0.1;
double Neuron::alpha = 0.5;

Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
{
	for (unsigned c = 0;c < numOutputs;c++)
	{
		m_outputWeights.push_back(Connection());
		m_outputWeights.back().weight = randomWeight();
	}
	m_myIndex = myIndex;
}
void Neuron::updateInputWeights(Layer& prevLayer)
{
	for (unsigned n = 0;n < prevLayer.size();++n)
	{
		Neuron& neuron = prevLayer[n];
		double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

		double newDeltaWeight =
			eta * neuron.getOutputVal() * m_gradient + alpha * oldDeltaWeight;
		neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
		neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
	}

}

double Neuron::sumDOW(const Layer& nextLayer)const
{
	double sum = 0.0;
	for (unsigned n = 0;n < nextLayer.size() - 1;++n)
	{
		sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
	}
	return sum;
}
void Neuron::calcHiddenGradients(const Layer& nextLayer)
{
	double dow = sumDOW(nextLayer);
	m_gradient = dow * Neuron::sigmoidDerivative(m_outputVal);
}
void Neuron::calcOutputGradients(double targetVal, int softm)
{
	double delta = targetVal - m_outputVal;
	double der = Neuron::softmaxDerivative(m_outputVal);
	if (softm)
	{
		m_gradient = delta * der;
	}
	else
	{
		m_gradient = delta * Neuron::sigmoidDerivative(m_outputVal);
	}


	//cout << m_gradient << " delta=" << delta << " derivative=" << der << endl;

}
double Neuron::sigmoid(double x)
{
	//return (1 / (1 + exp(-x)));
	//return sigmoid(x);
	//if (x > 0.0) return x;
	//else return 0.01 * x;
	return max(0.0, x);//relu
	//return 1 / (1 + exp(-x));//sigmoid
	//return tanh(x);
}
double Neuron::sigmoidDerivative(double x)
{
	//return (sigmoid(x) * (1 - sigmoid(x)));
	//if (x > 0.0) return 1;
	//else return -0.01;
	return double(x > 0.0);//relu
	//return transferFunction(x) * (1 - transferFunction(x));//sigmoid
	//return sigmoidDerivative(x);
}
double Neuron::softmax(double x, double constant)
{
	return exp(x - constant);
}

double Neuron::softmaxDerivative(double x)
{
	return (x * (1 - x));
}
void Neuron::feedForward(const Layer& prevLayer)
{
	double sum = 0.0;
	for (unsigned n = 0;n < prevLayer.size();++n)
	{
		sum += prevLayer[n].getOutputVal() * prevLayer[n].m_outputWeights[m_myIndex].weight;
	}
	m_outputVal = Neuron::sigmoid(sum);
}
void Neuron::ffsoftm(const Layer& prevLayer, double constant)
{
	double sum = 0.0;
	for (unsigned n = 0;n < prevLayer.size();++n)
	{
		sum += prevLayer[n].getOutputVal() * prevLayer[n].m_outputWeights[m_myIndex].weight;
	}
	m_outputVal = Neuron::softmax(sum, constant);

}
double Neuron::calcWeightedSum(Layer& prevLayer)
{
	double sum = 0.0;
	for (unsigned n = 0;n < prevLayer.size();++n)
	{
		sum += prevLayer[n].getOutputVal() * prevLayer[n].m_outputWeights[m_myIndex].weight;
	}
	return sum;

}




class Net
{
public:
	Net(const vector<unsigned>& topology);
	void feedForward(const vector<double>& inputVals, int softm);
	void backProp(const vector<double>& targetVals);
	void getResults(vector<double>& resultsVals) const;
	double getRecentAverageError(void)const { return m_recentAverageError; }
	double getRecentError(void)const { return m_error; }
	double calcSoftConst(int layerNum);

private:
	vector<Layer> m_layers;
	double m_error;
	double m_recentAverageError;
	double m_recentAverageSmoothingFactor;
};

void Net::getResults(vector<double>& resultsVals) const
{
	resultsVals.clear();
	resultsVals.reserve(m_layers.back().size());
	for (unsigned n = 0;n < m_layers.back().size() - 1;++n)
	{
		resultsVals.push_back(m_layers.back()[n].getOutputVal());
	}
}
void Net::backProp(const vector<double>& targetVals)
{
	Layer& outputLayer = m_layers.back();
	m_error = 0.0;
	for (unsigned n = 0;n < outputLayer.size() - 1;++n)
	{
		double delta = targetVals[n] - outputLayer[n].getOutputVal();
		m_error += delta * delta;
	}
	m_error /= outputLayer.size() - 1;
	m_error = sqrt(m_error);

	m_recentAverageError = (m_recentAverageError * m_recentAverageSmoothingFactor + m_error);
	for (unsigned n = 0;n < outputLayer.size() - 1;++n)
	{
		//cout << "Neuron" <<n<<" grad=";
		outputLayer[n].calcOutputGradients(targetVals[n], 1);

	}

	for (unsigned layerNum = m_layers.size() - 2;layerNum > 0;--layerNum)
	{
		Layer& hiddenLayer = m_layers[layerNum];
		Layer& nextLayer = m_layers[layerNum + 1];
		for (unsigned n = 0;n < hiddenLayer.size();++n)
		{
			hiddenLayer[n].calcHiddenGradients(nextLayer);
		}
	}
	for (unsigned layerNum = m_layers.size() - 1;layerNum > 0;--layerNum)
	{
		Layer& layer = m_layers[layerNum];
		Layer& prevLayer = m_layers[layerNum - 1];

		for (unsigned n = 0;n < layer.size() - 1;++n)
		{
			layer[n].updateInputWeights(prevLayer);
		}
	}
}
Net::Net(const vector<unsigned>& topology)
{
	unsigned numLayers = topology.size();
	for (unsigned layerNum = 0;layerNum < numLayers;layerNum++)
	{
		m_layers.push_back(Layer());
		unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

		for (unsigned neuronNum = 0;neuronNum <= topology[layerNum];neuronNum++)
		{
			m_layers.back().push_back(Neuron(numOutputs, neuronNum));
			cout << "Neuron" << neuronNum << "created" << endl;

		}
		m_layers.back().back().setOutputVal(0.0);//later
	}
}
void Net::feedForward(const vector<double>& inputVals, int softm)
{
	assert(inputVals.size() == m_layers[0].size() - 1);
	int limit = m_layers.size();
	//if (softm) --limit;

	for (unsigned i = 0;i < inputVals.size();++i)
	{
		m_layers[0][i].setOutputVal(inputVals[i]);
	}
	for (unsigned layerNum = 1;layerNum < limit;++layerNum)
	{
		Layer& prevLayer = m_layers[layerNum - 1];
		double constant = calcSoftConst(layerNum);/*for softmax*/
		for (unsigned n = 0;n < m_layers[layerNum].size() - 1;++n)
		{
			//m_layers[layerNum][n].feedForward(prevLayer);
			m_layers[layerNum][n].ffsoftm(prevLayer, constant);/**/
		}
	}
	/*if (softm)
	{
		int layerNum = m_layers.size() - 1;
		double constant=calcSoftConst(layerNum);

		for (unsigned n = 0;n < m_layers[layerNum].size() - 1;++n)
		{
			m_layers[layerNum][n].ffsoftm(m_layers[layerNum-1],constant);
		}

	}*/

}
double Net::calcSoftConst(int layerNum)
{
	;
	Layer& curLayer = m_layers[layerNum];
	Layer& prevLayer = m_layers[layerNum - 1];

	double constant = 0.0;
	double m = -INFINITY;
	double val;
	double sum = 0.0;
	vector<double> values;
	for (unsigned n = 0;n < curLayer.size() - 1;++n)
	{
		val = curLayer[n].calcWeightedSum(prevLayer);

		if (val > m) m = val;
		values.push_back(val);
	}

	for (int i = 0;i < curLayer.size() - 1;++i)
	{
		sum += exp(values[i] - m);
	}

	constant = m + log(sum);
	return constant;
}

void showVectorVals(string label, vector<double>& v)
{
	cout << label << " ";
	for (unsigned i = 0;i < v.size();++i)
	{
		cout << v[i] << " ";
	}
}
int main()
{

	TrainingData inputFile("train_images.txt");
	vector<double> inputVals, resultVals, targetVals;
	TrainingData outputFile("train_labels.txt");
	vector<unsigned> topology;
	topology.push_back(784);
	topology.push_back(512);
	topology.push_back(10);
	Net myNet(topology);

	int trainingPass = 0;
	cout << "Start of Training" << endl;
	auto start = std::chrono::system_clock::now();
	while (trainingPass != 50000) {
		++trainingPass;

		//cout << endl << "EPOCH " << trainingPass;

		if (inputFile.getNextInputs(inputVals) != topology[0]) {
			showVectorVals(" Inputs:", inputVals);
			break;
		}

		//showVectorVals(" Inputs:", inputVals);
		myNet.feedForward(inputVals, 1);

		myNet.getResults(resultVals);



		outputFile.getTargetOutputs(targetVals);

		myNet.backProp(targetVals);

		//assert(targetVals.size() == topology.back());
		if (trainingPass % 100 == 0)
		{
			cout << endl << "EPOCH " << trainingPass;

			showVectorVals(" Outputs:", resultVals);
			showVectorVals(" Targets:", targetVals);
			cout << "Net recent average error:" << myNet.getRecentAverageError() << endl;

		}




	}

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	cout << endl << "Training done.Time taken:" << elapsed_seconds.count() << endl;
	/*
	cout << "Starting Testing now" << endl;
	TrainingData tinputFile("test_images.txt");
	TrainingData toutputFile("train_labels.txt");
	int testPass = 0;
	while (testPass != 100) {
		++testPass;



		if (inputFile.getNextInputs(inputVals) != topology[0]) {
			//showVectorVals(" Inputs:", inputVals);
			break;
		}
		//showVectorVals(" Inputs:", inputVals);
		myNet.feedForward(inputVals);

		myNet.getResults(resultVals);


		outputFile.getTargetOutputs(targetVals);

		//assert(targetVals.size() == topology.back());

		if (testPass <=100)
		{
			cout << endl << "EPOCH " << testPass << endl;
			showVectorVals(" Outputs:", resultVals);cout << endl;
			showVectorVals(" Targets:", targetVals);cout << endl;
			cout << "Net recent average error:" << myNet.getRecentAverageError() << endl;
		}


	}
	//myNet.getResults(resultVals);
	//showVectorVals(" Outputs:", resultVals);
	//cout << "Net recent average error:" << myNet.getRecentAverageError() << endl;
	*/

}