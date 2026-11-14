#include "EmbeddingService.h"
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unistd.h>
#include <sys/wait.h>
#include <cstdio>
#include <cstdlib>

EmbeddingService::EmbeddingService() : python_in(nullptr), python_out(nullptr) {
    int pipe_to_python[2], pipe_from_python[2];
    
    if (pipe(pipe_to_python) == -1 || pipe(pipe_from_python) == -1) {
        throw std::runtime_error("Failed to create pipes");
    }
    
    pid_t pid = fork();
    if (pid == -1) {
        throw std::runtime_error("Failed to fork process");
    }
    
    if (pid == 0) {
        close(pipe_to_python[1]);   
        close(pipe_from_python[0]); 
        
        // 重定向stdin和stdout
        dup2(pipe_to_python[0], STDIN_FILENO); 
        dup2(pipe_from_python[1], STDOUT_FILENO); 
        
        close(pipe_to_python[0]);
        close(pipe_from_python[1]);
        
        const char* configured_python = std::getenv("SOSR_PYTHON");
        if (configured_python && configured_python[0] != '\0') {
            execl(configured_python, "python3", "./embedding_service.py", (char*)NULL);
        }

        execl("/usr/bin/python3", "python3", "./embedding_service.py", (char*)NULL);
        
        execl("/bin/python3", "python3", "./embedding_service.py", (char*)NULL);
        execl("/usr/local/bin/python3", "python3", "./embedding_service.py", (char*)NULL);
        
        execl("/usr/bin/python", "python", "./embedding_service.py", (char*)NULL);
        
        std::cerr << "ERROR: Failed to start Python process!" << std::endl;
        _exit(1);  
    } else {
        close(pipe_to_python[0]);    
        close(pipe_from_python[1]); 
        
        python_in = fdopen(pipe_to_python[1], "w");
        python_out = fdopen(pipe_from_python[0], "r");
        
        if (!python_in || !python_out) {
            throw std::runtime_error("Failed to open file descriptors");
        }
        
        std::cout << "Waiting for Python model to load..." << std::endl;
        
        sleep(4); 
        
        fprintf(python_in, "test\n");
        fflush(python_in);
        
        char test_buffer[50000];
        if (fgets(test_buffer, sizeof(test_buffer), python_out) != nullptr) {
            std::cout << "Python service communication verified!" << std::endl;
        } else {
            throw std::runtime_error("Failed to communicate with Python service - check if embedding_service.py exists");
        }
        
        std::cout << "Embedding service ready!" << std::endl;
    }
}

EmbeddingService::~EmbeddingService() {
    // 安全关闭管道
    if (python_in) {
        fprintf(python_in, "EXIT\n");  
        fflush(python_in);
        fclose(python_in);
        python_in = nullptr;
    }
    if (python_out) {
        fclose(python_out);
        python_out = nullptr;
    }
}

void EmbeddingService::getEmbeddings(const std::vector<std::string>& texts, std::vector<std::vector<double>> &results) {    
    if (!python_in || !python_out) {
        throw std::runtime_error("Python service not initialized");
    }
    
    std::string request;
    for (size_t i = 0; i < texts.size(); ++i) {
        if (i > 0) request += "|||";
        request += texts[i];
    }
    
    std::cout << "Sending: " << request << std::endl;
    
    fprintf(python_in, "%s\n", request.c_str());
    fflush(python_in);
    
    //std::vector<std::vector<double>> results;
    char buffer[50000];
    
    for (size_t i = 0; i < texts.size(); ++i) {
        if (fgets(buffer, sizeof(buffer), python_out) == nullptr) {
            throw std::runtime_error("Failed to read embedding from Python service");
        }
        
        std::vector<double> embedding;
        std::stringstream ss(buffer);
        std::string token;
        
        while (ss >> token) {
            try {
                embedding.push_back(std::stof(token));
            } catch (const std::exception& e) {
                throw std::runtime_error("Failed to parse embedding token: " + token);
            }
        }
        
        if (embedding.empty()) {
            throw std::runtime_error("Received empty embedding for text: " + texts[i]);
        }
        
        std::cout << "Got embedding " << (i+1) << "/" << texts.size() 
                  << " (" << embedding.size() << "D)" << std::endl;
        results.push_back(embedding);
    }
    
}
