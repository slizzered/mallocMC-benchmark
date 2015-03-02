#include <iostream>
#include <vector>
#include <sstream>
#include <string>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <boost/filesystem.hpp>
#include <map>




class benchmarkRun{
  std::vector<unsigned> allocationClocks;
  std::vector<unsigned> freeClocks;

  public:

  unsigned threads;
  benchmarkRun(unsigned nThreads):threads(nThreads){};
  benchmarkRun():threads(0){};

  void addLine(std::string line){
    std::istringstream iss(line);
    unsigned threadcount, allocation, free;
    if(!(iss >> threadcount >> allocation >> free)) {
      std::cout << "Error while parsing input: " << line << std::endl;
    }

    if(threads==0){
      threads = threadcount;
    }

    assert(threads == threadcount);

    allocationClocks.push_back(allocation);
    freeClocks.push_back(free);
  }

  std::string giveLine(){
    std::stringstream ss;
    ss << threads << "    ";
    ss << *std::min_element(allocationClocks.begin(),allocationClocks.end()) << "    ";
    ss << *std::max_element(allocationClocks.begin(),allocationClocks.end()) << "    ";
    ss << *std::min_element(freeClocks.begin(), freeClocks.end()) << "    ";
    ss << *std::max_element(freeClocks.begin(), freeClocks.end()) << "    ";
    ss << std::accumulate(allocationClocks.begin(),allocationClocks.end(),0)/allocationClocks.size() << "    ";
    ss << std::accumulate(freeClocks.begin(),freeClocks.end(),0)/freeClocks.size() << std::endl;
    return ss.str();
  }

};

inline bool operator==(const benchmarkRun& lhs, const benchmarkRun& rhs){ return lhs.threads == rhs.threads; }
inline bool operator!=(const benchmarkRun& lhs, const benchmarkRun& rhs){return !operator==(lhs,rhs);}
inline bool operator< (const benchmarkRun& lhs, const benchmarkRun& rhs){ return lhs.threads < rhs.threads; }
inline bool operator> (const benchmarkRun& lhs, const benchmarkRun& rhs){return  operator< (rhs,lhs);}
inline bool operator<=(const benchmarkRun& lhs, const benchmarkRun& rhs){return !operator> (lhs,rhs);}
inline bool operator>=(const benchmarkRun& lhs, const benchmarkRun& rhs){return !operator< (lhs,rhs);}


benchmarkRun parseSingleFile(std::string path){
  std::ifstream infile(path.c_str());
  std::string line;

  benchmarkRun run;

  while(std::getline(infile,line)){
    run.addLine(line);
  }

  return run;
}

void writeRunsToFile(std::string path, std::vector<benchmarkRun> runs){
  std::sort(runs.begin(), runs.end());
  std::ofstream ofile;
  ofile.open(path.c_str());
  ofile << "#threads   minAllocClocks    maxAllocClocks     minFreeClocks    maxFreeClocks    meanAllocClocks    meanFreeClocks\n";
  for(int i=0; i<runs.size(); ++i){
    ofile << runs.at(i).giveLine();
  }
  ofile.close();
}

int main(int argc, char* argv[]){
  std::string p(argc <= 1 ? "." : argv[1]);
  std::vector<benchmarkRun> runs;

  using namespace boost::filesystem;

  if (is_directory(p))
  {
    for (directory_iterator itr(p); itr!=directory_iterator(); ++itr)
    {
      std::cout << itr->path().filename() << ' '; // display filename only
      if (is_regular_file(itr->status())) std::cout << " [" << file_size(itr->path()) << ']';
      std::cout << '\n';

      runs.push_back(parseSingleFile((itr->path()).string()));
    }
  }
  else{
    runs.push_back(parseSingleFile(p));
  }

  std::string outDir = "";
  if(is_directory(p))
    outDir = p + "/";
    
  writeRunsToFile(outDir+"output.dat",runs);
  return 0;
}
