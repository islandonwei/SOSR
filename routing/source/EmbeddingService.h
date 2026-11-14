#ifndef EMBEDDING_SERVICE_H
#define EMBEDDING_SERVICE_H

#include <vector>
#include <string>
#include <cstdio>

class EmbeddingService {
private:
    FILE* python_in;  
    FILE* python_out; 
    
public:
    EmbeddingService();
    ~EmbeddingService();
    
    EmbeddingService(const EmbeddingService&) = delete;
    EmbeddingService& operator=(const EmbeddingService&) = delete;
    
    void getEmbeddings(const std::vector<std::string>& texts, std::vector<std::vector<double>> &queryEmb);
};

#endif // EMBEDDING_SERVICE_H
