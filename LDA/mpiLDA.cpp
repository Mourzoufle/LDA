/*
 * mpiLDA.cpp
 *
 *  Created on: 2012-8-31
 *      Author: Passerby
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include "mpi.h"

using namespace std;

/* Print a manual */
void manual() {
	cout << "Usage: [mpiexec -n [process number]] ./mpiLDA [options]" << endl;
	cout << "Options:" << endl;
	cout << "\t--test\t\t\t(Optional) Set program runs in test mode" << endl;
	cout << "\t--topic [integer]\tNumber of topics, must be positive, useless in test mode" << endl;
	cout << "\t--iteration [integer]\tMaximum iterations that will be done, must be positive" << endl;
	cout << "\t--alpha [float]\t\tHyper parameter ALPHA, must be positive" << endl;
	cout << "\t--beta [float]\t\tHyper parameter BETA, must be positive" << endl;
	cout << "\t--threshold [float]\t(Optional) The threshold of likelihood difference between iterations that suggests convergence" << endl;
	cout << "\t--data [string]\t\tDirectory of training or test data" << endl;
	cout << "\t--model [string]\tDirectory of resulting (in training mode) or existing(in test mode) model" << endl;
	cout << "\t--list [string]\t\tDirectory of result (list)" << endl;
	cout << "\t--matrix [string]\tDirectory of result (matrix)" << endl;
	cout << "\t--perplexity [string]\tDirectory of perplexity in each iteration" << endl;
	exit(0);
}

int main(int argc, char *argv[]) {
	bool test = false;				// [argument] test on existing model?
	int num_document;					// number of documents
	int num_topic = 0;				// [argument] number of topics
	int num_word = 0;					// number of words in vocabulary
	int num_word_model;				// number of words in model
	int num_list = 0;					// number of words in corpus on each process
	int num_list_all = 0;			// number of total words in corpus
	int num_iteration = 0;			// [argument] number of iterations
	int *list_length;					// lengths of each document
	int *list_document;				// document information of each word in corpus
	int *list_topic;					// topic information of each word in corpus
	int *list_word;					// words in corpus
	int *vector_topic;				// topic frequency over corpus
	int *matrix_document_topic;	// topic frequency over documents
	int *matrix_topic_word;			// word frequency over topics
	double alpha = 0;					// [argument] hyper parameter: ALPHA
	double beta = 0;					// [argument] hyper parameter: BETA
	double threshold = 0;			// [argument] threshold of convergence
	double base_likelihood;			// base likelihood of parameters
	string dir_data;					// [argument] directory of training/test data
	string dir_model;					// [argument] directory of model (written when trained and read when tested)
	string dir_result_matrix;		// [argument] directory of matrix in result
	string dir_result_list;			// [argument] directory of list in result
	string dir_perplexity;			// [argument] directory of perplexity in each iteration
	int id;								// [MPI configuration] ID of process in MPI
	int num_process;					// [MPI configuration] total number of processes
	int *send;							// [MPI configuration] send buffer
	int *recv;							//	[MPI configuration] receive buffer

	srand((unsigned)time(NULL));

	// setup MPI environment...
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &num_process);
	MPI_Barrier(MPI_COMM_WORLD);
	double start = MPI_Wtime();
	string prefix;
	stringstream stream;
	stream << id;
	stream >> prefix;
	prefix += "_";
	stream.clear();

	// parse command line...
	for (int i = 1; i < argc; i++) {
		string command = argv[i];
		if (!command.compare("--test")) {
			test = true;
			num_topic = 0;
		}
		else if (i == argc - 1)
			break;
		else if (!command.compare("--topic") && !test)
			num_topic = atoi(argv[++i]);
		else if (!command.compare("--iteration"))
			num_iteration = atoi(argv[++i]);
		else if (!command.compare("--alpha"))
			alpha = atof(argv[++i]);
		else if (!command.compare("--beta"))
			beta = atof(argv[++i]);
		else if (!command.compare("--threshold"))
			threshold = atof(argv[++i]);
		else if (!command.compare("--data"))
			dir_data = argv[++i];
		else if (!command.compare("--model"))
			dir_model = argv[++i];
		else if (!command.compare("--matrix"))
			dir_result_matrix = prefix + argv[++i];
		else if (!command.compare("--list"))
			dir_result_list = prefix + argv[++i];
		else if (!command.compare("--perplexity"))
			dir_perplexity = argv[++i];
	}

	// check command line...
	if (id == 0) {
		if (!test && (num_topic <= 0))
			manual();
		if ((num_iteration <= 0) || (alpha <= 0) || (beta <= 0))
			manual();
	}

	// test input&output files...
	ifstream file_data(dir_data.c_str());
	if (!file_data.is_open()) {
		cout << "Failed in opening data file " << dir_data << endl;
		exit(0);
	}
	ofstream file_result_matrix(dir_result_matrix.c_str());
	if (!file_result_matrix.is_open()) {
		cout << "Failed in opening result matrix file " << dir_result_matrix << endl;
		exit(0);
	}
	ofstream file_result_list(dir_result_list.c_str());
	if (!file_result_list.is_open()) {
		cout << "Failed in opening result list file " << dir_result_list << endl;
		exit(0);
	}
	ofstream file_perplexity(dir_perplexity.c_str());
	if ((id == 0) && !file_perplexity.is_open()) {
		cout << "Failed in opening perplexity file " << dir_perplexity << endl;
		exit(0);
	}
	fstream file_model;
	if (id == 0) {
		if (test)
			file_model.open(dir_model.c_str(), ios::in);
		else
			file_model.open(dir_model.c_str(), ios::out);
		if (!file_model.is_open()) {
			cout << "Failed in opening model file " << dir_model << endl;
			exit(0);
		}

		// CHECK POINT
		cout << "Initialization success!" << endl;
		if (!test)
			cout <<"Training..." << endl;
		else
			cout <<"Testing..." << endl;
		cout << "\tIteration number: " << num_iteration << endl;
		if (threshold > 0)
			cout << "\tLikelihood threshold: " << threshold << endl;
		cout << "\tALPHA: " << alpha << endl;
		cout << "\tBETA: " << beta << endl;
	}

	// read data...
	int tmp1, tmp2;
	vector<int> lengths;
	vector<int*> documents;
	file_data >> tmp1;
	int index = 0;
	while (!file_data.eof()) {
		int *document;
		bool read = index++ % num_process == (unsigned)id;
		if (read) {
			num_list += tmp1;
			lengths.push_back(tmp1);
			document = new int[tmp1];
			documents.push_back(document);
		}
		for (int i = 0; i < tmp1; i++) {
			num_list_all++;
			file_data >> tmp2;
			if (!read)
				continue;
			if (num_word <= tmp2)
				num_word = tmp2 + 1;
			document[i] = tmp2;
		}
		file_data >> tmp1;
	}
	file_data.close();

	// read model if testing...
	if (test && (id == 0)) {
		file_model >> num_topic >> num_word_model;
		num_word = num_word > num_word_model ? num_word : num_word_model;
	}

	// synchronize number of topics, words and total words...
	send = new int[2];
	recv = new int[2];
	send[0] = num_topic;
	send[1] = num_word;
	MPI_Allreduce(send, recv, 2, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
	num_topic = recv[0];
	num_word = recv[1];
	delete[] send;
	delete[] recv;

	// build data sets...
	num_document = lengths.size();
	list_length = new int[num_document];
	for (int i = 0; i < num_document; i++)
		list_length[i] = lengths[i];
	lengths.clear();
	list_document = new int[num_list];
	list_word = new int[num_list];
	index = 0;
	for (int i = 0; i < num_document; i++) {
		for (int j = 0; j < list_length[i]; j++) {
			list_word[index] = documents[i][j];
			list_document[index++] = i;
		}
		delete[] documents[i];
	}
	documents.clear();
	int word, topic, document;
	int size_data = num_document * num_topic;		// size of document-topic matrix
	int size_model = num_topic * num_word;			// size of topic-word matrix
	int size_vector = sizeof(int) * num_topic;	// memory size of topic distribution vector
	int size_matrix = sizeof(int) * size_model;	// memory size of topic-word matrix
	list_topic = new int[num_list];
	vector_topic = new int[num_topic];
	matrix_document_topic = new int[size_data];
	matrix_topic_word = new int[size_model];
	memset(vector_topic, 0, size_vector);
	memset(matrix_document_topic, 0, sizeof(int) * size_data);
	memset(matrix_topic_word, 0, size_matrix);
	for (int i = 0; i < num_list; i++) {
		document = list_document[i];
		topic = rand() % num_topic;
		list_topic[i] = topic;
		matrix_document_topic[document * num_topic + topic]++;
		if (!test) {
			word = list_word[i];
			vector_topic[topic]++;
			matrix_topic_word[topic * num_word + word]++;
		}
	}

	// move on reading model if testing...
	if (test && (id == 0)) {
		int tmp;
		int offset = 0;
		for (int i = 0; i < num_topic; i++) {
			for (int j = 0; j < num_word_model; j++) {
				file_model >> tmp;
				vector_topic[i] += tmp;
				matrix_topic_word[offset++] = tmp;
			}
		}
		file_model.close();
	}

	// CHECK POINT
	if (id == 0) {
		cout << "Read input success!" << endl;
		cout << "\tTopic number: " << num_topic << endl;
		cout << "\tWord number: " << num_word << endl;
		cout << "\tTotal words: " << num_list_all << endl;
	}

	// synchronize topic-word matrix and topic vector...
	recv = new int[size_model];
	MPI_Allreduce(matrix_topic_word, recv, size_model, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	memcpy(matrix_topic_word, recv, size_matrix);
	MPI_Allreduce(vector_topic, recv, num_topic, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	memcpy(vector_topic, recv, size_vector);

	// generate base likelihood...
	double part_likelihood = 0;
	for (int i = 0; i < num_document; i++)
		part_likelihood += list_length[i] * log(1.0 * list_length[i] / num_list_all);
	MPI_Allreduce(&part_likelihood, &base_likelihood, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	// generate an order...
	int *order = new int[num_list];
	for (int i = 0; i < num_list; i++)
		order[i] = i;
	for (int i = 0; i < num_list - 1; i++) {
		int r = i + rand() % (num_list - i);
		order[i] ^= order[r];
		order[r] ^= order[i];
		order[i] ^= order[r];
	}

	int *difference = new int[size_model];
	double topic_alpha = alpha * num_topic;
	double word_beta = beta * num_word;
	double *probability = new double[num_topic];	// posterior probability distribution
	double *vector_likelihood = new double[num_iteration];

	// perform Gibbs sampling iterations...
	if (id == 0)
		cout << "Sampling start..." << endl;
	for (int r = 0; r < num_iteration; r++) {
		memset(difference, 0, sizeof(int) * size_model);
		for (int i = 0; i < num_list; i++) {
			int index = order[i];
			document = list_document[index];
			topic = list_topic[index];
			word = list_word[index];
			int offset = document * num_topic;
			matrix_document_topic[offset + topic]--;
			if (!test) {
				vector_topic[topic]--;
				difference[topic * num_word + word]--;
			}

			// calculating posterior probability...
			double sum = 0;
			for (int j = 0; j < num_topic; j++) {
				probability[j] = (alpha + matrix_document_topic[offset + j]) * (beta + matrix_topic_word[j * num_word + word] + difference[j * num_word + word]) / (word_beta + vector_topic[j]);
				sum += probability[j];
			}

			// sample a topic...
			sum *= 1.0 * rand() / RAND_MAX;
			topic = -1;
			while(++topic < num_topic - 1) {
				if (sum > probability[topic])
					sum -= probability[topic];
				else
					break;
			}
			list_topic[index] = topic;
			matrix_document_topic[offset + topic]++;
			if (!test) {
				vector_topic[topic]++;
				difference[topic * num_word + word]++;
			}
		}

		// synchronize topic-word matrix and topic vector...
		if (!test) {
			MPI_Allreduce(difference, recv, size_model, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
			memcpy((int *)difference, recv, size_matrix);
			memset(vector_topic, 0, size_vector);
			int offset = 0;
			for (int i = 0; i < num_topic; i++) {
				for (int j = 0; j < num_word; j++) {
					matrix_topic_word[offset] += difference[offset];
					vector_topic[i] += matrix_topic_word[offset++];
				}
			}
		}

		// compute likelihood...
		double likelihood = 0;
		for (int i = 0; i < num_list; i++) {
			document = list_document[i];
			word = list_word[i];
			int offset = document * num_topic;
			double sum = 0;
			for (int j = 0; j < num_topic; j++)
				sum += (alpha + matrix_document_topic[offset + j]) * (beta + matrix_topic_word[j * num_word + word]) / ((topic_alpha + list_length[document]) * (word_beta + vector_topic[j]));
			likelihood += log(sum);
		}
		MPI_Allreduce(&likelihood, &vector_likelihood[r], 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		if ((r > 0) && (fabs(vector_likelihood[r] - vector_likelihood[r - 1]) < threshold))
			num_iteration = r + 1;
		if (id == 0)
			cout << "Iteration = " << r + 1 << "\tLog of likelihood = " << fixed << vector_likelihood[r] + base_likelihood << "\tPerplexity = " << exp(-vector_likelihood[r] / num_list_all) << endl;
	}

	if (id == 0) {
		// output perplexity...
		for (int i = 0; i < num_iteration; i++)
			file_perplexity << fixed << exp(-vector_likelihood[i] / num_list_all) << endl;
		cout << "Write perplexity file success!" << endl;

		// output model if training...
		if (!test) {
			file_model << "topic: " << num_topic << " word: " << num_word << endl;
			int offset = 0;
			for (int i = 0; i < num_topic; i++) {
				for (int j = 0; j < num_word - 1; j++)
					file_model << matrix_topic_word[offset++] << " ";
				file_model << matrix_topic_word[offset++] << endl;
			}
			file_model.close();
			cout << "Write model file success!" << endl;
		}
	}

	// output result matrix...
	file_result_matrix << "document: " << num_document << " topic" << num_topic << endl;
	int offset = 0;
	for (int i = 0; i < num_document; i++) {
		for (int j = 0; j < num_topic - 1; j++)
			file_result_matrix << matrix_document_topic[offset++] << " ";
		file_result_matrix << matrix_document_topic[offset++] << endl;
	}
	file_result_matrix.close();

	// output result list...
	offset = 0;
	for (int i = 0; i < num_document; i++) {
		int num = list_length[i];
		file_result_list << num;
		for (int j = 0; j < num - 1; j++)
			file_result_list << " " << list_topic[offset++];
		file_result_list << " " << list_topic[offset++] << endl;
	}
	file_result_list.close();

	MPI_Barrier(MPI_COMM_WORLD);
	double end = MPI_Wtime();
	if (id == 0)
		cout << "Computing time: " << end - start << endl;
	MPI_Finalize();
}
