#pragma once
class CSVDataReader
{
public:
	CSVDataReader(std::string dir, int, int);
	float** getDataMtx() { return data; }
	float* getData(int);

private:
	float** data;
};