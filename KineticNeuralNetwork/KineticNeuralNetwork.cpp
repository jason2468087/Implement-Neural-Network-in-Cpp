// KineticNeuralNetwork.cpp : 此檔案包含 'main' 函式。程式會於該處開始執行及結束執行。
//

#include <iostream>
#include <math.h>
#include <array>
#include <vector>
#include <iomanip>
#include <string>
#include <fstream>
#include <sstream>
#include <time.h>
#include <stdint.h>
#include <immintrin.h>


#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"


// ~~~~~Network Setting~~~~~

const int NETWORK_DEPTH = 3;
int LAST_LAYER = NETWORK_DEPTH - 1;
int NETWORK_STRUCTURE[NETWORK_DEPTH] = {784,300 ,10};
int TRANSFER_FUNCTION_TYPE[NETWORK_DEPTH] = {1,1,0 };
float LEARNING_RATE[NETWORK_DEPTH] = {0.01,0.01,0.5};
int INPUT_SIZE = NETWORK_STRUCTURE[0];
int OUTPUT_SIZE = NETWORK_STRUCTURE[LAST_LAYER];

int TRAINING_EXTRACT_SIZE = 60000;
int TESTING_EXTRACT_SIZE = 10000;

int TRAIN_ITER = 10000;
int TEST_ITER = 100;
int TEST_PERIOD = 100;
int BATCH_SIZE = 1;

float RAND_W_MIN = -0.1;
float RAND_W_MAX = 0.1;

bool outputIsLabel = true;

const std::string ROOT_DIR = "C:\\Users\\jason\\OneDrive\\圖片\\Machine Learning Data\\MNIST";

// ~~~~~Timer~~~~~
clock_t csvTime = 0;
clock_t imgTime = 0;
clock_t setupTime = 0;
clock_t forwardTime = 0;
clock_t inputTime = 0;
clock_t dotTime = 0;
clock_t actTime = 0;
clock_t backTime = 0;
clock_t gradTime = 0;
clock_t zGradTime = 0;
clock_t wGradTime = 0;
clock_t aGradTime = 0;
clock_t refreshTime = 0;
clock_t errorTime = 0;
clock_t tempStart = 0;
clock_t tempStart1 = 0;
clock_t tempStart2 = 0;

// ~~~~~Utility Functions~~~~~
void printArr(float* arr,int s)
{
	for (int i = 0; i < s; i++)
	{
		std::cout << std::setprecision(3) << arr[i] <<" "; //
	}
	std::cout << std::endl;
}

void printArr(int* arr, int s)
{
	for (int i = 0; i < s; i++)
	{
		std::cout << std::setprecision(3) << arr[i] << " "; //
	}
	std::cout << std::endl;
}

void printImg(float* arr, int r, int c)
{
	std::cout << std::fixed << std::setprecision(0);
	for (int n = 0; n < r*c; n++)
	{
		if (n % c == 0)
		{
			std::cout << std::endl;
		}
		std::cout << arr[n]*1 << " ";
	}
}

void printMtx(float **arr, int r, int c)
{
	for (int j = 0; j < r; j++)
	{
		for (int i = 0; i < c; i++)
		{
			std::cout << std::setprecision(3) << arr[j][i] << " ";
		}
		std::cout << std::endl;
	}
}

float* getMtxColumn(float** mtx,int rLength,int cIdx)
{
	float* col = new float[rLength];
	int s_10 = rLength / 10 * 10;
	int n = 0;

	if (rLength >= 10)
	{
		for (n = n; n < s_10; n += 10)
		{
			col[n] = mtx[n][cIdx];
			col[n+1] = mtx[n+1][cIdx];
			col[n+2] = mtx[n+2][cIdx];
			col[n+3] = mtx[n+3][cIdx];
			col[n+4] = mtx[n+4][cIdx];
			col[n+5] = mtx[n+5][cIdx];
			col[n+6] = mtx[n+6][cIdx];
			col[n+7] = mtx[n+7][cIdx];
			col[n+8] = mtx[n+8][cIdx];
			col[n+9] = mtx[n+9][cIdx];
		}
	}

	for (n = n; n < rLength; n++)
	{
		col[n] = mtx[n][cIdx];
	}

	return col;
}

float dot2(float* a, float* b,int s)
{
	float dp = 0;

	for (int i = 0; i < s ; i++)
	{
		dp += a[i] * b[i];
	}
	return dp;
}

float dot(float* a, float* b, int s)
{
	float dp = 0.0f;
	int n = 0;
	int s_8 = s / 8 * 8;
	__m256 vsum = _mm256_setr_ps(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
	__m256 va, vb, vs;

	for (n = n; n < s_8; n += 8)
	{
		va = _mm256_loadu_ps(&a[n]);
		vb = _mm256_loadu_ps(&b[n]);

		vs = _mm256_mul_ps(va, vb);
		vsum = _mm256_add_ps(vsum, vs);
	}

	float* vsum_ = (float*)&vsum;
	dp = vsum_[0] + vsum_[1] + vsum_[2] + vsum_[3] + vsum_[4] + vsum_[5] + vsum_[6] + vsum_[7];

	for (n = n; n < s; n++)
	{
		dp += a[n] * b[n];
	}

	return dp;
}

float dotV(float** a, float* b,int c, int s)
{
	float dp = 0;
	int n = 0;
	int s_16 = s/16*16;

	if (s >= 16)
	{
		for (n = n; n < s_16; n += 16)
		{
			//dp += a[n][c]*b[n]+a[n+1][c]*b[n+1]+a[n+2][c]*b[n+2]+a[n+3][c]*b[n+3]+a[n+4][c]*b[n+4]+a[n+5][c]*b[n+5]+a[n+6][c]*b[n+6]+a[n+7][c]*b[n+7]+a[n+8][c]*b[n+8]+a[n+9][c]*b[n+9];
			dp += a[n][c]*b[n]+a[n+1][c]*b[n+1]+a[n+2][c]*b[n+2]+a[n+3][c]*b[n+3]+a[n+4][c]*b[n+4]+a[n+5][c]*b[n+5]+a[n+6][c]*b[n+6]+a[n+7][c]*b[n+7]
				+a[n+8][c]*b[n+8]+a[n+9][c]*b[n+9]+a[n+10][c]*b[n+10]+a[n+11][c]*b[n+11]+a[n+12][c]*b[n+12]+a[n+13][c]*b[n+13]+a[n+14][c]*b[n+14]+a[n+15][c]*b[n+15];
		}
	}
	
	for (n = n; n < s; n++)
	{
		dp += a[n][c] * b[n];
	}
	return dp;
}

float avg(float* arr,int s)
{
	float sum = 0;

	for (int i = 0; i < s; i++)
	{
		sum += arr[i];
	}

	return sum/s;
}

int maxIdx(float* arr, int s)
{
	int max = 0;

	for (int i = 0 ; i < s ; i++)
	{
		if (arr[i] > arr[max])
		{
			max = i;
		}
	}

	return max;
}

float transFunc(float z, int type)
{
	if (type == 0)
	{
		return 1.0f / (1.0f + exp(-z));
	}
	else
	{
		if (z >= 0)
		{
			return z;
		}
		else
		{
			return 0.01f*z;
		}
	}
	
	//
}

float dTransFunc(float z, int type)
{
	if (type == 0)
	{
		float tf = transFunc(z, type);
		return tf * (1.0f - tf);
	}
	else
	{
		if (z >= 0)
		{
			return 1.0f;
		}
		else
		{
			return 0.01f;
		}
	}
	//
}

float optimizer_(float w, float g,float lr)
{
	return w-lr*g;
}

void optimizer(float* w, float* g, float lr, int s)
{
	int n = 0;
	int s_8 = s / 8 * 8;
	__m256 vLr = _mm256_setr_ps(lr, lr, lr, lr, lr, lr, lr, lr);
	__m256 vw, vg, vlrg;
	float* vw_;
	for (n = n; n < s_8; n += 8)
	{
		vw = _mm256_loadu_ps(&w[n]);
		vg = _mm256_loadu_ps(&g[n]);

		vlrg = _mm256_mul_ps(vLr, vg);
		vw = _mm256_sub_ps(vw, vlrg);

		vw_ = (float*)&vw;
		w[n] = vw_[0];
		w[n+1] = vw_[1];
		w[n+2] = vw_[2];
		w[n+3] = vw_[3];
		w[n+4] = vw_[4];
		w[n+5] = vw_[5];
		w[n+6] = vw_[6];
		w[n+7] = vw_[7];
	}

	for (n = n; n < s; n++)
	{
		w[n] = w[n] - lr * g[n];
	}
}

// ~~~~~CSV Data Reader~~~~~
class CSVDataReader
{
public:
	CSVDataReader(std::string dir, int, int,bool);
	float** getDataMtx() { return data; }
	float* getData(int);

private:
	float** data;
};

CSVDataReader::CSVDataReader(std::string dir, int eSize, int dSize, bool isLabel)
{
	clock_t csvTempStart = clock();
	std::ifstream file;
	file.open(dir);
	std::string line;
	int extractSize = eSize;
	int dataSize = dSize;

	data = new float*[extractSize];
	for (int i = 0; i < extractSize; ++i)
	{
		data[i] = new float[dataSize];
	}

	for (int i = 0; i < extractSize; i++)
	{
		for (int j = 0; j < dataSize; j++)
		{
			data[i][j] = 0.0f;
		}
	}

	if (!isLabel)
	{
		int i = 0; int j = 0;

		file.seekg(0, file.end);
		int fileSize = file.tellg();
		file.seekg(0, file.beg);

		char* buffer = new char[fileSize];

		file.read(buffer, fileSize);

		int n = 0; 
		int numSize = 0;		

		while(i < extractSize)
		{
			if (buffer[n] != ',' && buffer[n] != '\n')
			{
				numSize++;
			}
			else
			{
				float floatNum = 0;

				for (int m = 0 ; m < numSize ; m++)
				{
					floatNum += (float)(buffer[n - m - 1] - '0') * pow(10.0f,(float)m);
				}
				data[i][j] = floatNum / 255.0f;
				numSize = 0;
				
				j++;
				
				if (j == INPUT_SIZE)
				{
					j = 0;
					i++;

					if (i % 500 == 0)
					{
						std::cout << "\r" << std::fixed << std::setprecision(2) << (float)i / extractSize * 100.0f << "%";
					}
				}
			}
			n++;
		}
	}
	else
	{
		for (int n = 0; n < extractSize; n++)
		{
			std::getline(file, line);
			data[n][(int)std::stof(line)] = 1.0f;
		}
	}
}

float* CSVDataReader::getData(int row)
{
	return data[row];
}

// ~~~~~Image Reader~~~~~
float* readImg(const char* dir)
{
	int width, height, channels;
	float* image = stbi_loadf(dir, &width, &height, &channels, 1);

	return image;
}

// ~~~~~NeuronLayer~~~~~

class NeuronLayer
{
public:
	NeuronLayer();
	void setupLayer(int, int, int,int,float);
	void calculateA(float*);
	float* calculateGrad(float*,float*, int);
	void refreshW();
	void setA(float*);
	void setW(float**);
	void setDEDA(float*);
	float* getA();
	float** getW();
	float* getDEDA();
	float* getDEDZ();
	float** getDEDW();

	//private:
	int layerType = 0;
	float lr;
	int s;
	int x_s;
	int bias;
	int layerNum;
	float* z;
	float* a;
	float** w;
	float* deda;
	float* dedz;
	float** dedw;
};
NeuronLayer::NeuronLayer()
{

};

void NeuronLayer::setupLayer(int s, int x_s, int lyNum, int layerType,float lrate)//
{
	this->layerNum = lyNum;
	this->layerType = layerType;
	this->lr = lrate;

	s = s + 1;
	x_s = x_s + 1;
	bias = s - 1;

	this->s = s;
	this->x_s = x_s;

	// Initiate Empty Arrays
	a = new float[s];
	z = new float[s];

	w = new float*[s];
	for (int i = 0; i < s; ++i)
	{
		w[i] = new float[x_s];
	}

	dedw = new float*[s];
	for (int i = 0; i < s; ++i)
	{
		dedw[i] = new float[x_s];
	}

	deda = new float[s];
	dedz = new float[s];

	// Assign initial value to array
	for (int i = 0; i < s; i++) // Loop y
	{
		z[i] = 0;
		a[i] = 0;

		for (int j = 0; j < x_s; j++) // Loop x
		{
			w[i][j] = ((float(rand()) / float(RAND_MAX)) * (RAND_W_MAX - (RAND_W_MIN))) + (RAND_W_MIN);
			dedw[i][j] = 0;
		}
	}
	a[bias] = 1;
};

void NeuronLayer::calculateA(float* x)  
{
	for (int i = 0; i < s-1; i++)// For every neuron except last(Bias) in current layer => calculate activation
	{
		tempStart1 = clock();
		z[i] = dot(w[i], x, x_s);
		dotTime += clock() - tempStart1;
		tempStart1 = clock();
		a[i] = transFunc(z[i], layerType);
		actTime += clock() - tempStart1;
	}
};

float* NeuronLayer::calculateGrad(float* prev_deda,float* prev_a, int lyNum)
{
	for (int i = 0; i < s - 1; i++) /// For every neuron except last(Bias) in current layer => Calculate z Gradient and all w Gradient
	{
		tempStart2 = clock();
		dedz[i] = dTransFunc(z[i], layerType)*deda[i]; //a[i] * (1 - a[i])
		zGradTime += clock() - tempStart2;

		tempStart2 = clock();
		int n = 0;
		int s_8 = x_s / 8 * 8;
		__m256 vdedz = _mm256_setr_ps(dedz[i], dedz[i], dedz[i], dedz[i], dedz[i], dedz[i], dedz[i], dedz[i]);
		__m256 vpa, vdedw;
		float* vdedw_;
		for (n = n; n < s_8; n += 8)
		{
			vpa = _mm256_loadu_ps(&prev_a[n]);

			vdedw = _mm256_mul_ps(vpa, vdedz);

			vdedw_ = (float*)&vdedw;
			dedw[i][n] = vdedw_[0];
			dedw[i][n + 1] = vdedw_[1];
			dedw[i][n + 2] = vdedw_[2];
			dedw[i][n + 3] = vdedw_[3];
			dedw[i][n + 4] = vdedw_[4];
			dedw[i][n + 5] = vdedw_[5];
			dedw[i][n + 6] = vdedw_[6];
			dedw[i][n + 7] = vdedw_[7];
		}

		for (n = n; n < x_s; n++)
		{
			dedw[i][n] = prev_a[n] * dedz[i];
		}
		wGradTime += clock() - tempStart2;
	}

	// Initiate prev_deda
	tempStart2 = clock();

	for (int j = 0; j < x_s; j++) // For every prev layer neuron => prev a Gradient = dot product(one connected w from every this layer neuron, all z gradients in this layer) 
	{
		prev_deda[j] = dotV(w, dedz, j, s - 1);
	}
	aGradTime += clock() - tempStart2;
	
	return prev_deda; // return prev a gradient for prev layer to obtain
}

void NeuronLayer::refreshW()
{
	for (int i = 0; i < s - 1; i++) // For every neuron except last(Bias)
	{
		optimizer(w[i], dedw[i], lr, x_s);
	}
}

void NeuronLayer::setA(float* A)
{
	a = A;
}

void NeuronLayer::setW(float **W)
{
	w = W;
}

void NeuronLayer::setDEDA(float* DEDA)
{
	delete[] deda;
	deda = DEDA;
}

float* NeuronLayer::getA()
{
	return a;
}

float** NeuronLayer::getW()
{
	return w;
}

float* NeuronLayer::getDEDA()
{
	return deda;
}

float* NeuronLayer::getDEDZ()
{
	return dedz;
}

float** NeuronLayer::getDEDW()
{
	return dedw;
}


// ~~~~~Network~~~~~
class Network
{
public:
	Network(int* );
	void setInput(float*);
	float* getOutput();
	float* getError();
	void feedForward();
	void backPropagation();
	void calculateE(float*);
	void calculateDEDY(float*,int);
	void calculateBatchDEDY();
	NeuronLayer* getLayerArr();

private:
	NeuronLayer *layerArr;
	float* y;
	float* e;
	float* dedy_; // temp dedy
	float** dedy; // a batch of gradients
	float* d;
};

Network::Network(int* netSize)
{
	layerArr = new NeuronLayer[NETWORK_DEPTH];
	for (int i = 0 ; i < NETWORK_DEPTH ; i++) // For every Neuronlayer from input to output => setup layer
	{
		if (i == 0)
		{
			layerArr[i].setupLayer(NETWORK_STRUCTURE[i], 0, i, TRANSFER_FUNCTION_TYPE[i], LEARNING_RATE[i]);
		}
		else
		{
			layerArr[i].setupLayer(NETWORK_STRUCTURE[i], NETWORK_STRUCTURE[i-1], i, TRANSFER_FUNCTION_TYPE[i], LEARNING_RATE[i]);
		}
	}

	// Declare and initialize arrays
	y = new float[OUTPUT_SIZE]; 
	e = new float[OUTPUT_SIZE];
	d = new float[OUTPUT_SIZE];
	dedy_ = new float[OUTPUT_SIZE];

	dedy = new float*[BATCH_SIZE];
	for (int b = 0; b < BATCH_SIZE; ++b)
	{
		dedy[b] = new float[OUTPUT_SIZE];
	}
};

void Network::setInput(float* input)
{
	tempStart1 = clock();
	layerArr[0].setA(input);
	layerArr[0].getA()[INPUT_SIZE] = 1;
	inputTime += clock() - tempStart1;
}

float* Network::getOutput()
{
	return y;
}

float* Network::getError()
{
	return e;
}

void Network::feedForward()
{
	for (int i = 1; i < NETWORK_DEPTH; i++) // For every Neuronlayer from 1st hinden to output layer => calculate layer activiations
	{
		layerArr[i].calculateA(layerArr[i-1].getA());
	}

	for (int i = 0 ; i < OUTPUT_SIZE ; i++) // For every element in output array => get its value
	{
		y[i] = layerArr[LAST_LAYER].getA()[i];
	}
}

void Network::calculateE(float* d)
{
	this->d = d;

	for (int i = 0; i < OUTPUT_SIZE; i++) // For every error array element => find error
	{
		e[i] = (y[i] - d[i])*(y[i] - d[i]) / 2;
	}
}

void Network::calculateDEDY(float* d,int b)
{
	this->d = d;

	for (int i = 0; i < OUTPUT_SIZE; i++) // For every gradient array element => find gradient
	{
		dedy[b][i] = y[i] - d[i];
	}
}

void Network::calculateBatchDEDY()
{
	// initialize array
	float* batchDEDY; 
	batchDEDY = new float[OUTPUT_SIZE];
	for (int i = 0; i < OUTPUT_SIZE; i++)
	{
		batchDEDY[i] = 0;
	}

	for (int i = 0 ; i < OUTPUT_SIZE; i++) // For every output neuron
	{
		for (int b = 0 ; b < BATCH_SIZE ; b++) // For every feedforward result in a batch of an output neuron => Sum up
		{
			batchDEDY[i] += dedy[b][i];
		}
		batchDEDY[i] /= BATCH_SIZE;
	}
	layerArr[LAST_LAYER].setDEDA(batchDEDY);
}

void Network::backPropagation()
{
	for (int i = LAST_LAYER; i >= 1; i--) // For every Neuronlayer from output to 1st hidden layer => calculate z gradient, w gradient and prev a gradient
	{
		tempStart1 = clock();
		layerArr[i].calculateGrad(layerArr[i - 1].getDEDA(),layerArr[i-1].getA(),i); //calculate z gradient, w gradient and return prev a gradient
		gradTime += clock() - tempStart1;

		tempStart1 = clock();
		layerArr[i].refreshW(); // This layer refresh weight
		refreshTime += clock() - tempStart1;
	}
}

NeuronLayer* Network::getLayerArr()
{
	return layerArr;
}

void printResult(Network net, CSVDataReader outputDataReader,int idx)
{
	std::cout << "X:" << std::endl;
	printArr(net.getLayerArr()[0].getA(), 4);
	std::cout << "W2:" << std::endl;
	printMtx(net.getLayerArr()[1].getW(), 5, 4);
	std::cout << "A2:" << std::endl;
	printArr(net.getLayerArr()[1].getA(), 6);
	std::cout << "W3:" << std::endl;
	printMtx(net.getLayerArr()[2].getW(), 2, 6);
	std::cout << "Y:" << std::endl;
	printArr(net.getOutput(), 2);

	std::cout << "Ans:" << std::endl;
	printArr(outputDataReader.getData(idx), 2);
	std::cout << "E:" << std::endl;
	printArr(net.getError(), 2);

	std::cout << "DEDY:" << std::endl;
	printArr(net.getLayerArr()[2].getDEDA(), 2);
	std::cout << "DEDZ3:" << std::endl;
	printArr(net.getLayerArr()[2].getDEDZ(), 2);
	std::cout << "DEDW3:" << std::endl;
	printMtx(net.getLayerArr()[2].getDEDW(), 2, 6);
	std::cout << "DEDA2:" << std::endl;
	printArr(net.getLayerArr()[1].getDEDA(), 5);
	std::cout << "DEDZ2:" << std::endl;
	printArr(net.getLayerArr()[1].getDEDZ(), 5);
	std::cout << "DEDW2:" << std::endl;
	printMtx(net.getLayerArr()[1].getDEDW(), 5, 4);
	std::cout << "DEDX:" << std::endl;
	printArr(net.getLayerArr()[0].getDEDA(), 3);
	std::cout << "W2:" << std::endl;
	printMtx(net.getLayerArr()[1].getW(), 5, 4);
	std::cout << "W3:" << std::endl;
	printMtx(net.getLayerArr()[2].getW(), 2, 6);
}

// ~~~~~Main Function~~~~~
int main()
{
	clock_t totalStart = clock();

	tempStart = clock();
	std::cout << "Extracting Training Data"<<std::endl;
	CSVDataReader trainDataReader(ROOT_DIR + "//MNIST train data.csv", TRAINING_EXTRACT_SIZE, INPUT_SIZE, !outputIsLabel);
	std::cout << "Extracting Training Label" << std::endl; 
	CSVDataReader trainLabelReader(ROOT_DIR + "//MNIST train label.csv", TRAINING_EXTRACT_SIZE, OUTPUT_SIZE, outputIsLabel); //ROOT_DIR+"MNIST train label.csv" //"C:\\Users\\jason\\OneDrive\\文件\\NNW Data Base\\BooleanOutput.csv"
	std::cout << "Extracting Testing Data" << std::endl; 
	CSVDataReader testDataReader(ROOT_DIR + "//MNIST test data.csv", TESTING_EXTRACT_SIZE, INPUT_SIZE, !outputIsLabel);
	std::cout << "Extracting Testing Label" << std::endl;
	CSVDataReader testLabelReader(ROOT_DIR + "//MNIST test label.csv", TESTING_EXTRACT_SIZE, OUTPUT_SIZE, outputIsLabel);
	csvTime += clock()- tempStart;

	tempStart = clock();
	std::cout << "Setup Network" << std::endl;
	Network net = Network(NETWORK_STRUCTURE);
	setupTime += clock() - tempStart;

	float* input = new float[3];
	input[0] = 1.0f; input[1] = 0.0f; input[2] = 1.0f; 

	float* ans = new float[3]; 
	ans[0] = 1.0f; ans[1] = 0.0f;

	std::cout << "Start Training" << std::endl;
	for (int i = 0; i < TRAIN_ITER; i++) // Training Loop
	{
		for (int b = 0 ; b < BATCH_SIZE ; b++) // Mini-Batch Loop
		{
			int randIdx = rand() % TRAINING_EXTRACT_SIZE; // Generate random Index

			tempStart = clock();
			//std::string TRAIN_IMG_DIR = ROOT_DIR + "//MNIST train data//" + std::to_string(randIdx) + ".jpg";
			//float* img = readImg(TRAIN_IMG_DIR.c_str());
			float* img = trainDataReader.getData(randIdx);
			imgTime += clock() - tempStart;

			tempStart = clock();
			net.setInput(img); //net.setInput(input); //

			net.feedForward();
			forwardTime += clock() - tempStart;

			net.calculateDEDY(trainLabelReader.getData(randIdx), b);//net.calculateDEDY(ans, b); //
		}

		// Progresional Testing
		if (i % TEST_PERIOD == 0)
		{
			int accCount = 0;
			for (int n = 0; n < TEST_ITER; n++) // Test Loop
			{
				int randIdx = rand() % TESTING_EXTRACT_SIZE; // Generate random Index

				tempStart = clock();
				//std::string TEST_IMG_DIR = ROOT_DIR + "//MNIST test data//" + std::to_string(randIdx) + ".jpg";
				//float* img = readImg(TEST_IMG_DIR.c_str());
				float* img = testDataReader.getData(randIdx);
				imgTime += clock() - tempStart;

				tempStart = clock();
				net.setInput(img);
				net.feedForward();
				forwardTime += clock() - tempStart;

				tempStart = clock();
				net.calculateE(testLabelReader.getData(randIdx));
				if (maxIdx(net.getOutput(), OUTPUT_SIZE) == maxIdx(testLabelReader.getData(randIdx), OUTPUT_SIZE))
				{
					accCount++;
				}
				errorTime += clock() - tempStart;
			}
			std::cout << "Iter:" << i <<"/"<<TRAIN_ITER<< " Accuracy:" << (float)accCount / (float)TEST_ITER*100.0 << "%" << std::endl;
		}

		net.calculateBatchDEDY();

		tempStart = clock();
		net.backPropagation();
		backTime += clock() - tempStart;
	}

	clock_t totalTime = clock()-totalStart;
	
	std::cout << "~~~Read CSV Version~~~" << std::endl;
	std::cout << "Total Time:" << totalTime / 1000.0<<"s" << std::endl << std::endl;
	std::cout << "Setup Time:            " << setupTime/1000.0 << "s   (" << (float)setupTime / totalTime * 100 << "%)" << std::endl;
	std::cout << "CSV Reader Time:       " << csvTime / 1000.0 << "s   (" << (float)csvTime / totalTime * 100 << "%)" << std::endl;
	std::cout << "Image Read Time:       " << imgTime / 1000.0 <<"s   ("<< (float)imgTime/totalTime*100 <<"%)" << std::endl;
	std::cout << "Feed Forward Time:     " << forwardTime / 1000.0 << "s   (" << (float)forwardTime / totalTime * 100 << "%)" << std::endl;
	std::cout << "     Set Input Time:        " << inputTime / 1000.0 << "s   (" << (float)inputTime / forwardTime * 100 << "%)" << std::endl;
	std::cout << "     Dot Product Time:      " << dotTime / 1000.0 << "s   (" << (float)dotTime / forwardTime * 100 << "%)" << std::endl;
	std::cout << "     Activation Time:       " << actTime / 1000.0 << "s   (" << (float)actTime / forwardTime * 100 << "%)" << std::endl;
	std::cout << "Back Propagation Time: " << backTime / 1000.0 << "s   (" << (float)backTime / totalTime * 100 << "%)" << std::endl;
	std::cout << "     Gradient Time:         " << gradTime / 1000.0 << "s   (" << (float)gradTime / backTime * 100 << "%)" << std::endl;
	std::cout << "          zGradient Time:        " << zGradTime / 1000.0 << "s   (" << (float)zGradTime / gradTime * 100 << "%)" << std::endl;
	std::cout << "          wGradient Time:        " << wGradTime / 1000.0 << "s   (" << (float)wGradTime / gradTime * 100 << "%)" << std::endl;
	std::cout << "          aGradient Time:        " << aGradTime / 1000.0 << "s   (" << (float)aGradTime / gradTime * 100 << "%)" << std::endl;
	std::cout << "     Refresh Time:          " << refreshTime / 1000.0 << "s   (" << (float)refreshTime / backTime * 100 << "%)" << std::endl;

	
/*int size = 45;
float** mtx = new float*[size];
float* arr1 = new float[size];
float* dp = new float[size];

for (int i = 0; i < size; i++) // Loop y
{
	arr1[i] = ((float(rand()) / float(RAND_MAX)) * (RAND_W_MAX - (RAND_W_MIN))) + (RAND_W_MIN);
	dp[i] = 0;
	mtx[i] = new float[size];

	for (int j = 0; j < size; j++)
	{
		mtx[i][j] = ((float(rand()) / float(RAND_MAX)) * (RAND_W_MAX - (RAND_W_MIN))) + (RAND_W_MIN);
	}
}

int iter = 1;
float f = 200.0f; char c;
int j = 0;
float data = 0;
std::ifstream file;
file.open(ROOT_DIR + "//MNIST train data.csv");

tempStart = clock();
for (int n = 0; n < iter; n++)
{
	for (int j = 0; j < size; j++) // For every prev layer neuron => prev a Gradient = dot product(one connected w from every this layer neuron, all z gradients in this layer) 
	{
		float sum = 0;
		for (int i = 0; i < size; i++) // For every current layer neuron except last(bias)
		{
			sum = sum + mtx[i][j] * arr1[i];
		}
		dp[j] = sum;
	}
	printArr(dp,size);
}
clock_t time1 = clock() - tempStart;

tempStart = clock();
for (int n = 0; n < iter; n++)
{
	for (int j = 0; j < size; j++) // For every prev layer neuron => prev a Gradient = dot product(one connected w from every this layer neuron, all z gradients in this layer) 
	{
		dp[j] = dotV(mtx,arr1,j,size);
	}
	printArr(dp, size);
}
clock_t time2 = clock() - tempStart;

tempStart = clock();
for (int n = 0; n < iter; n++)
{
	//optimizer2(arr1, arr2, 0.1f, size);
}
clock_t time3 = clock() - tempStart;

std::cout << time1 << " " << time2 << " " << time3;*/

	return 0;
}

/*for (int n = 55000 ; n < 55020 ; n++)
	{
		//printImg(trainDataReader.getData(n),28,28);
	}*

	/*int iter = 1000000;
	float f = 200.0f; char c;
	int j = 0;
	float data = 0;
	std::ifstream file;
	file.open(ROOT_DIR + "//MNIST train data.csv");

	tempStart = clock();
	for (int n = 0 ; n < iter; n++)
	{
		data = f / 255.0f;
	}
	clock_t time1 = clock() - tempStart;

	tempStart = clock();
	for (int n = 0; n < iter; n++)
	{
		file >> f;
		file.ignore(1);
	}
	clock_t time2 = clock() - tempStart;

	tempStart = clock();
	for (int n = 0; n < iter; n++)
	{
		file >> f;
		//file.get();
	}
	clock_t time3 = clock() - tempStart;

	std::cout << time1 << " " << time2<<" "<<time3;*/