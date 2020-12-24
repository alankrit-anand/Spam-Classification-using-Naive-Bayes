%%cu

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iterator>
#include <algorithm>
#include <set>
#include <map>

#include <cstdlib>
#include <ctime>
#include <cmath>
#include <stdint.h>
#include <chrono>
typedef std::chrono::high_resolution_clock Clock;

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

/**********************************************************/
/******************** Kernel Functions ********************/
/**********************************************************/


__global__ void multiply(double* input){

	const int tid = threadIdx.x;

	auto step_size = 1;
	int number_of_threads = blockDim.x;

	while (number_of_threads > 0){
		if (tid < number_of_threads) {
			const auto fst = tid * step_size * 2;
			const auto snd = fst + step_size;
			input[fst] *= input[snd];
		}

		step_size <<= 1; 
		number_of_threads >>= 1;
	}
}



/**********************************************************/
/******************** Global Variables ********************/
/**********************************************************/

fstream test_file;

string *word_vector;
double *spam_vector;
double *ham_vector;

int spam_length = 0;
int ham_length = 0;

struct item{
    int spam_occurence = 0;
    int ham_occurence = 0;
};

map<string, item> mp;
set<string> heap;


/**********************************************************/
/******************** GPU functions ********************/
/**********************************************************/

int dim = 1024;

void print(double p_test_spam, double p_test_ham){
   
    double p_spam = (double) spam_length / (spam_length + ham_length);
	double p_ham = (double) ham_length / (spam_length + ham_length);
    double p_spam_test = p_test_spam * p_spam;
    double p_ham_test = p_test_ham * p_ham;
    p_spam_test /= p_spam_test + p_ham_test;

    cout << "Probabilty of being spam: " << p_spam_test << endl;
    cout << "Probabilty of being ham: " << 1 - p_spam_test << endl;

    if (p_spam_test > 0.5) {
		cout << "Message is a spam." << endl;
	}
	else {
		cout << "Message is not a spam." << endl;
	}
}

void run_gpu() {
	double p_test_spam = 1.0;
	double p_test_ham = 1.0;
	int size = 1024 * sizeof(double);
    double *d1, *d2; 
    cudaMalloc(&d1, size);
    cudaMalloc(&d2, size);

    for(int st = 0; st < dim; st += 1024){
        cudaMemcpy(d1, spam_vector + st, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d2, ham_vector + st, size, cudaMemcpyHostToDevice);
        multiply <<<1, 512>>>(d1);
        multiply <<<1, 512>>>(d2);
        double r1, r2; 
        cudaMemcpy(&r1, d1, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(&r2, d2, sizeof(double), cudaMemcpyDeviceToHost);
        p_test_spam *= r1;
        p_test_ham *= r2;
    }
	print(p_test_spam, p_test_ham);
}


void run_cpu() {
	double p_test_spam = 1.0;
	double p_test_ham = 1.0;

    for(int i=0; i<dim; i++){
        p_test_spam *= spam_vector[i];
        p_test_ham *= ham_vector[i];
    }

	print(p_test_spam, p_test_ham);
}


/**********************************************************/
/*********************** Functions ************************/
/**********************************************************/

bool isCharacter(char &c){
    if(c >= 'a' && c<='z')
        return 1;
    if(c >= 'A' && c<='Z'){
        c += 'a' - 'A';
        return 1;
    }
    return 0;
}






void run_naive_bayes() {

    while(1){
        string s;
        test_file >> s;
        if(s=="")
            break;
        string tmp;
        for(char &c : s){
            if(isCharacter(c))
                tmp.push_back(c);
        }
        s = tmp;
        heap.insert(s);
    }

    while(dim < (int)mp.size())
        dim *= 2;
    
    for(int i=0; i<(int)mp.size(); i++){
        string word = word_vector[i];
        if(!heap.count(word)){
            spam_vector[i] = 1 - spam_vector[i];
            ham_vector[i] = 1 - ham_vector[i];
        }
    }

    for(int i=(int)mp.size(); i<dim; i++){
        spam_vector[i] = ham_vector[i] = 1;
    }
}


/**********************************************************/
/****************** Initializing Model ********************/
/**********************************************************/

enum class CSVState {
    UnquotedField,
    QuotedField,
    QuotedQuote
};


std::vector<std::string> readCSVRow(const std::string &row) {
    CSVState state = CSVState::UnquotedField;
    std::vector<std::string> fields {""};
    size_t i = 0; // index of the current field
    for (char c : row) {
        switch (state) {
            case CSVState::UnquotedField:
                switch (c) {
                    case ',': // end of field
                              fields.push_back(""); i++;
                              break;
                    case '"': state = CSVState::QuotedField;
                              break;
                    default:  fields[i].push_back(c);
                              break; }
                break;
            case CSVState::QuotedField:
                switch (c) {
                    case '"': state = CSVState::QuotedQuote;
                              break;
                    default:  fields[i].push_back(c);
                              break; }
                break;
            case CSVState::QuotedQuote:
                switch (c) {
                    case ',': // , after closing quote
                              fields.push_back(""); i++;
                              state = CSVState::UnquotedField;
                              break;
                    case '"': // "" -> "
                              fields[i].push_back('"');
                              state = CSVState::QuotedField;
                              break;
                    default:  // end of quote
                              state = CSVState::UnquotedField;
                              break; }
                break;
        }
    }
    return fields;
}


std::vector<std::vector<std::string>> readCSV(std::istream &in) {
    std::vector<std::vector<std::string>> table;
    std::string row;
    while (!in.eof()) {
        std::getline(in, row);
        if (in.bad() || in.fail()) {
            break;
        }
        auto fields = readCSVRow(row);
        table.push_back(fields);
    }
    return table;
}




vector<string> split(string str){
    str += ' ';
    vector<string> res;
    string temp;
    for(int i=0; i<(int)str.size(); i++){
        if(isCharacter(str[i]))
            temp.push_back(str[i]);
        else if(temp.size())
            res.push_back(temp), temp.clear();
    }
    return res;
}


void finish_vectors() {

    int n = mp.size();
    spam_vector = new double[n];
    ham_vector = new double[n];
    word_vector = new string[n];

    auto it = mp.begin();
    for(int i=0; i<n; i++, it++){
        word_vector[i] = it->first;
        
        spam_vector[i] = (double) max(1, it->second.spam_occurence) / spam_length;
        ham_vector[i] = (double) max(1, it->second.ham_occurence) / ham_length;
        
    }

}


void create_words_vector(vector<vector<string>> &table) {
    
    for(vector<string> &v : table){
        
        vector<string> tokens = split(v[1]);
        
        if(v[0] == "spam"){
            for(string s : tokens){
                mp[s].spam_occurence++;
            }
             spam_length++;
        } 
        if(v[0] == "ham"){
            for(string s : tokens){
                mp[s].ham_occurence++;
                
            }
            ham_length++;
        } 
    }
	
    

	finish_vectors();
}


/**********************************************************/
/**************** Execution begins HERE *******************/
/**********************************************************/


int main(){
    
    
    ifstream file("spam.csv");
    test_file.open("test.txt", ios::in);
    vector<vector<string>> table = readCSV(file);
    cout << "******************* Naive Bayes Classifier for Spam Detection ********************" << endl << endl;

    

    auto c1 = Clock::now();
    create_words_vector(table);
    auto c2 = Clock::now();
    cout << "Initialization of Naive Bayes Model Completed in: " << chrono::duration_cast<std::chrono::milliseconds>(c2 - c1).count() << " milliseconds" << endl << endl << endl;
    
    
    
    c1 = Clock::now();
    cout << "CPU Execution began:" << endl;
    run_naive_bayes();
    run_cpu();
    c2 = Clock::now();
    cout << "CPU execution Completed in: " << chrono::duration_cast<chrono::microseconds>(c2 - c1).count() << " microseconds" << endl << endl << endl;

    c1 = Clock::now();
    cout << "GPU Execution began:" << endl;
    run_gpu();
    c2 = Clock::now();
    cout << "GPU execution Completed in: " << chrono::duration_cast<chrono::microseconds>(c2 - c1).count() << " microseconds" << endl;

    return 0;
}