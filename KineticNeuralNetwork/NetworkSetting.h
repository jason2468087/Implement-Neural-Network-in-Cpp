#pragma once
float alpha = 0.5f;  //learning rate

 extern const int networkDepth = 4;
int networkStructure[networkDepth + 1] = { 0, 3, 8,8 ,2 };
int inputSize = networkStructure[1];
int outputSize = networkStructure[networkDepth];

int trainingExtractSize = 8;
int testingExtractSize = 8;

int trainIter = 50000;
int testIter = 8;
int batchSize = 8;