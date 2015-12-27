#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <iomanip>

using namespace std;

/* Print a manual */
void manual() {
	cout << "./LDA [options]" << endl;
	cout << "Options:" << endl;
	cout << "\t--test\t\t\t(Optional) Set program runs in test mode" << endl;
	cout << "\t--topic [integer]\tNumber of topics, must be positive, useless in test mode" << endl;
	cout << "\t--iteration [integer]\tMaximum iterations that will be done, must be positive" << endl;
	cout << "\t--alpha [float]\t\tHyper parameter ALPHA, must be positive" << endl;
	cout << "\t--beta [float]\t\tHyper parameter BETA, must be positive" << endl;
	cout << "\t--threshold [float]\t(Optional) The threshold of likelihood difference between iterations that suggests convergence" << endl;
	cout << "\t--data [string]\t\tDirectory of training or test data" << endl;
	cout << "\t--model [string]\tDirectory of resulting (in training mode) or existing(in test mode) model" << endl;
	cout << "\t--result [string]\t\tDirectory of result" << endl;
	exit(0);
}

int main(int argc, char *argv[]) {
	bool test = false;				// [argument] test on existing model?
	int num_document;					// number of documents
	int num_topic = 0;				// [argument] number of topics
	int num_word = 0;					// number of words in vocabulary
	int num_word_model;				// number of words in model
	int num_list = 0;					// number of words in corpus
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
	double base_likelihood = 0;		// base likelihood of parameters
	string dir_data;					// [argument] directory of training/test data
	string dir_model;					// [argument] directory of model
	string dir_result;				// [argument] directory of result


	srand((unsigned)time(NULL));

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
		else if (!command.compare("--result"))
			dir_result = argv[++i];
	}

	// check command line...
	if (!test && (num_topic <= 0))
		manual();
	if ((num_iteration <= 0) || (alpha <= 0) || (beta <= 0))
		manual();

	// test input&output files...
	ifstream file_data(dir_data);
	if (!file_data.is_open()) {
		cout << "Failed in opening data file " << dir_data << endl;
		exit(0);
	}
	ifstream file_model(dir_model);
	if (test && !file_model.is_open()) {
		cout << "Failed in opening data file " << dir_model << endl;
		exit(0);
	}
	ofstream file_result(dir_result);
	if (!file_result.is_open()) {
		cout << "Failed in opening result file " << dir_result << endl;
		exit(0);
	}

	// CHECK POINT
	cout << "Initialization success!" << endl;
	if (!test)
		cout << "Training..." << endl;
	else
		cout << "Testing..." << endl;
	cout << "\tIteration number: " << num_iteration << endl;
	if (threshold > 0)
		cout << "\tLikelihood threshold: " << threshold << endl;
	cout << "\tALPHA: " << alpha << endl;
	cout << "\tBETA: " << beta << endl;

	// read data...
	int tmp1, tmp2;
	vector<int> lengths;
	vector<int*> documents;
	file_data >> tmp1;
	int index = 0;
	while (!file_data.eof()) {
		int *document;
		num_list += tmp1;
		lengths.push_back(tmp1);
		document = new int[tmp1];
		documents.push_back(document);
		for (int i = 0; i < tmp1; i++) {
			file_data >> tmp2;
			if (num_word <= tmp2)
				num_word = tmp2 + 1;
			document[i] = tmp2;
		}
		file_data >> tmp1;
	}
	file_data.close();

	// read model if testing...
	if (test) {
		file_model >> num_topic >> num_word_model;
		num_word = num_word > num_word_model ? num_word : num_word_model;
	}

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
	int size_document_topic = num_document * num_topic;		// size of document-topic matrix
	int size_topic_word = num_topic * num_word;			// size of topic-word matrix
	list_topic = new int[num_list];
	vector_topic = new int[num_topic];
	matrix_document_topic = new int[size_document_topic];
	matrix_topic_word = new int[size_topic_word];
	memset(vector_topic, 0, sizeof(int) * num_topic);
	memset(matrix_document_topic, 0, sizeof(int) * size_document_topic);
	memset(matrix_topic_word, 0, sizeof(int) * size_topic_word);
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
	if (test) {
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
	cout << "Read input success!" << endl;
	cout << "\tTopic number: " << num_topic << endl;
	cout << "\tWord number: " << num_word << endl;
	cout << "\tTotal words: " << num_list << endl;

	// generate base likelihood...
#pragma omp parallel for reduction(+ : base_likelihood)
	for (int i = 0; i < num_document; i++)
		base_likelihood += list_length[i] * log(1.0 * list_length[i] / num_list);

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

	double topic_alpha = alpha * num_topic;
	double word_beta = beta * num_word;
	double *probability = new double[num_topic];	// posterior probability distribution
	double *vector_likelihood = new double[num_iteration];

	// perform Gibbs sampling iterations...
	cout << "Sampling start..." << endl;
	for (int r = 0; r < num_iteration; r++) {
		for (int i = 0; i < num_list; i++) {
			int index = order[i];
			document = list_document[index];
			topic = list_topic[index];
			word = list_word[index];
			int offset = document * num_topic;
			matrix_document_topic[offset + topic]--;
			if (!test) {
				vector_topic[topic]--;
				matrix_topic_word[topic * num_word + word]--;
			}

			// calculating posterior probability...
			double sum = 0;
			for (int j = 0; j < num_topic; j++) {
				probability[j] = (alpha + matrix_document_topic[offset + j]) * (beta + matrix_topic_word[j * num_word + word] + matrix_topic_word[j * num_word + word]) / (word_beta + vector_topic[j]);
				sum += probability[j];
			}

			// sample a topic...
			sum *= 1.0 * rand() / RAND_MAX;
			topic = -1;
			while (++topic < num_topic - 1) {
				if (sum > probability[topic])
					sum -= probability[topic];
				else
					break;
			}
			list_topic[index] = topic;
			matrix_document_topic[offset + topic]++;
			if (!test) {
				vector_topic[topic]++;
				matrix_topic_word[topic * num_word + word]++;
			}
		}

		// compute likelihood...
		vector_likelihood[r] = 0;
		for (int i = 0; i < num_list; i++) {
			document = list_document[i];
			word = list_word[i];
			int offset = document * num_topic;
			double sum = 0;
			for (int j = 0; j < num_topic; j++)
				sum += (alpha + matrix_document_topic[offset + j]) * (beta + matrix_topic_word[j * num_word + word]) / ((topic_alpha + list_length[document]) * (word_beta + vector_topic[j]));
			vector_likelihood[r] += log(sum);
		}
		if ((r > 0) && (fabs(vector_likelihood[r] - vector_likelihood[r - 1]) < threshold))
			num_iteration = r + 1;
		cout << "Iteration = " << r + 1 << "\tLog of likelihood = " << fixed << vector_likelihood[r] + base_likelihood << "\tPerplexity = " << exp(-vector_likelihood[r] / num_list) << endl;
	}

	// output result...
	int offset = 0;
	for (int i = 0; i < num_document; i++) {
		int topic = 0;
		int max = matrix_document_topic[offset++];
		for (int j = 1; j < num_topic; j++) {
			if (matrix_document_topic[offset] > max) {
				max = matrix_document_topic[offset];
				topic = j;
			}
			offset++;
		}
		file_result << topic << endl;
	}
	ofstream output((dir_result + "phi.txt").c_str());
	for (int i = 0; i < num_topic; i++) {
		for (int j = 0; j < num_word; j++)
			output << matrix_topic_word[i * num_word + j] << " ";
		output << endl;
	}
	output.close();
	output.open((dir_result + "theta.txt").c_str());
	for (int i = 0; i < num_document; i++) {
		for (int j = 0; j < num_topic; j++)
			output << matrix_document_topic[i * num_topic + j] << " ";
		output << endl;
	}
	output.close();
}
