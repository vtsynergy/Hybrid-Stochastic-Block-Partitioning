#include "utils.hpp"

#include "mpi_data.hpp"

std::string utils::build_filepath() {
    std::ostringstream filepath_stream;
    filepath_stream << args.directory << "/" << args.type << "/" << args.overlap << "Overlap_" << args.blocksizevar;
    filepath_stream << "BlockSizeVar/" << args.type << "_" << args.overlap << "Overlap_" << args.blocksizevar;
    filepath_stream << "BlockSizeVar_" << args.numvertices << "_nodes";
    // TODO: Add capability to process multiple "streaming" graph parts
    std::string filepath = filepath_stream.str();
    if (!fs::exists(filepath + ".tsv")) {
        std::cerr << "File doesn't exist: " << filepath + ".tsv" << std::endl;
        exit(-1);
    }
    return filepath;
}

std::vector<std::vector<std::string>> utils::read_csv(fs::path &filepath) {
    std::vector<std::vector<std::string>> contents;
    if (!fs::exists(filepath)) {
        if (mpi.rank == 0)
            std::cerr << "File doesn't exist: " << filepath << std::endl;
        return contents;
    }
    std::string line;
    std::ifstream file(filepath);
    int items = 0;
    while (std::getline(file, line)) {
        std::vector<std::string> row;
        std::stringstream line_stream(line);
        std::string value;
        while (line_stream >> value) {
            row.push_back(value);
            items++;
        }
        contents.push_back(row);
    }
    if (mpi.rank == 0)
        std::cout << "Read in " << contents.size() << " lines and " << items << " values." << std::endl;
    return contents;
}

void utils::insert(NeighborList &neighbors, int from, int to) {
    if (from >= (int)neighbors.size()) {
        std::vector<std::vector<int>> padding(from - neighbors.size() + 1, std::vector<int>());
        neighbors.insert(neighbors.end(), padding.begin(), padding.end());
    }
    neighbors[from].push_back(to);
}

void utils::insert_nodup(NeighborList &neighbors, int from, int to) {
    if (from >= (int)neighbors.size()) {
        std::vector<std::vector<int>> padding(from - neighbors.size() + 1, std::vector<int>());
        neighbors.insert(neighbors.end(), padding.begin(), padding.end());
    }
    for (const int &neighbor : neighbors[from])
        if (neighbor == to) return;
    neighbors[from].push_back(to);
}

bool utils::insert(std::unordered_map<int, int> &map, int key, int value) {
    std::pair<std::unordered_map<int, int>::iterator, bool> result = map.insert(std::make_pair(key, value));
    return result.second;
}
