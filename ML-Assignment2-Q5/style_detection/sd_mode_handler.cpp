#include "sd_mode_handler.h"
#include "sd_data_loader.h"
#include "create_network.h"
#include "random.h"
#include "sgf_loader.h"
#include <memory>
#include <vector>
#include <fstream>
#include <iostream>

namespace strength_detection {

using namespace minizero;

SDModeHandler::SDModeHandler()
{
    RegisterFunction("sgf_policy", this, &SDModeHandler::runSGFPolicy);
    RegisterFunction("test_sgf_loader", this, &SDModeHandler::testSGFLoader);
}

void SDModeHandler::runSGFPolicy()
{
    std::shared_ptr<network::Network> network = network::createNetwork(config::nn_file_name, 3);
    std::shared_ptr<network::AlphaZeroNetwork> az_network = std::static_pointer_cast<network::AlphaZeroNetwork>(network);
    std::vector<EnvironmentLoader> env_loaders = loadEnvironmentLoaders();

    for (auto& env_loader : env_loaders) {
        Environment env;
        for (auto& action_pair : env_loader.getActionPairs()) {
            az_network->pushBack(env.getFeatures());
            env.act(action_pair.first);
        }

        std::vector<std::shared_ptr<network::NetworkOutput>> network_outputs = az_network->forward();
        for (auto& output : network_outputs) {
            std::shared_ptr<network::AlphaZeroNetworkOutput> az_output = std::static_pointer_cast<network::AlphaZeroNetworkOutput>(output);
            for (auto& p : az_output->policy_) { std::cerr << p << " "; }
            std::cerr << std::endl;
        }
    }
}

void SDModeHandler::testSGFLoader()
{
    std::cerr << "=== Testing SGF Loader ===" << std::endl;
    std::vector<EnvironmentLoader> env_loaders = loadEnvironmentLoaders();
    
    std::cerr << "\n=== Summary ===" << std::endl;
    std::cerr << "Total games loaded: " << env_loaders.size() << std::endl;
    
    for (size_t i = 0; i < env_loaders.size() && i < 3; ++i) {
        std::cerr << "\nGame " << (i+1) << ":" << std::endl;
        std::cerr << "  Black: " << env_loaders[i].getTag("PB") << std::endl;
        std::cerr << "  White: " << env_loaders[i].getTag("PW") << std::endl;
        std::cerr << "  Result: " << env_loaders[i].getTag("RE") << std::endl;
        std::cerr << "  Board Size: " << env_loaders[i].getTag("SZ") << std::endl;
        std::cerr << "  Komi: " << env_loaders[i].getTag("KM") << std::endl;
        std::cerr << "  Total Moves: " << env_loaders[i].getActionPairs().size() << std::endl;
    }
    
    if (env_loaders.size() > 3) {
        std::cerr << "\n... and " << (env_loaders.size() - 3) << " more games" << std::endl;
    }
}

std::vector<EnvironmentLoader> SDModeHandler::loadEnvironmentLoaders()
{
    std::vector<EnvironmentLoader> env_loaders;

    // Read SGF file
    std::string sgf_file = strength_detection::sgf_file_path;
    if (sgf_file.empty()) {
        std::cerr << "Error: sgf_file_path is not set in configuration" << std::endl;
        return env_loaders;
    }

    std::ifstream fin(sgf_file, std::ifstream::in);
    if (!fin.is_open()) {
        std::cerr << "Error: Cannot open SGF file: " << sgf_file << std::endl;
        return env_loaders;
    }

    std::cerr << "Reading SGF file: " << sgf_file << std::endl;

    // Skip the first line (player name)
    std::string player_name;
    std::getline(fin, player_name);
    std::cerr << "Player: " << player_name << std::endl;

    // Read each SGF game - using shared parsing logic
    int game_count = 0;
    for (std::string content; std::getline(fin, content);) {
        EnvironmentLoader env_loader;
        if (parseSGFToEnvironmentLoader(content, env_loader)) {
            env_loaders.push_back(env_loader);
            game_count++;
        } else {
            std::cerr << "Warning: Failed to parse SGF at line " << (game_count + 2) << std::endl;
        }
    }

    std::cerr << "Loaded " << game_count << " games from " << sgf_file << std::endl;
    return env_loaders;
}

} // namespace strength_detection
