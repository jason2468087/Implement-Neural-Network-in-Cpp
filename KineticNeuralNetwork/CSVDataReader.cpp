#include "CSVDataReader.h"

CSVDataReader::CSVDataReader(std::string dir, int eSize, int dSize)
{
	std::fstream file;
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
			data[i][j] = 0;
		}
	}

	std::cout << "Extracting Data:" << std::endl;
	for (int n = 0; n < extractSize; n++)
	{
		std::cout << "\r" << std::fixed << std::setprecision(2) << (float)(n + 1) / extractSize * 100.0 << "%"; if (n == extractSize - 1) { std::cout << std::endl; }

		getline(file, line, '\n');
		std::istringstream templine(line);
		std::string unitData;

		int i = 0;
		while (getline(templine, unitData, ','))
		{
			data[n][i] = std::stof(unitData);
			i++;
		}
	}
}

float* CSVDataReader::getData(int row)
{
	return data[row];
}