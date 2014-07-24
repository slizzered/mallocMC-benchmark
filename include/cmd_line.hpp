#pragma once

#include <iostream>
#include <sstream>
#include <vector>
#include <utility>

// define some defaults
static const unsigned threads_default = 128;
static const unsigned blocks_default  = 64; 
static const size_t heapInMB_default  = 1024; // 1GB



/**
 *  * prints a helpful message about program use
 *   *
 *    * @param argv the argv-parameter from main, used to find the program name
 *     */
void print_help(char** argv){
  std::stringstream s;

  s << "SYNOPSIS:"                                              << std::endl;
  s << argv[0] << " [OPTIONS]"                                  << std::endl;
  s << ""                                                       << std::endl;
  s << "OPTIONS:"                                               << std::endl;
  s << "  -h, --help"                                           << std::endl;
  s << "    Print this help message and exit"                   << std::endl;
  s << ""                                                       << std::endl;
  s << "  -v, --verbose"                                        << std::endl;
  s << "    Print information about parameters and progress"    << std::endl;
  s << ""                                                       << std::endl;
  s << "  -m, --machine_readable"                               << std::endl;
  s << "    Print all relevant parameters as CSV. This will"    << std::endl;
  s << "    suppress all other output unless explicitly"        << std::endl;
  s << "    requested with --verbose or -v"                     << std::endl;
  s << ""                                                       << std::endl;
  s << "  --threads=N"                                          << std::endl;
  s << "    Set the number of threads per block (default "                  ;   
  s <<                               threads_default << "128)"  << std::endl;
  s << ""                                                       << std::endl;
  s << "  --blocks=N"                                           << std::endl;
  s << "    Set the number of blocks in the grid (default "                 ;   
  s <<                                   blocks_default << ")"  << std::endl;
  s << ""                                                       << std::endl;
  s << "  --heapsize=N"                                         << std::endl;
  s << "    Set the heapsize to N Megabyte (default "                       ;   
  s <<                         heapInMB_default << "1024)"      << std::endl;

  std::cout << s.str();
}

/**
 *  * will parse command line arguments
 *   *
 *    * for more details, see print_help()
 *     *
 *      * @param argc argc from main()
 *       * @param argv argv from main()
 *        * @param heapInMP will be filled with the heapsize, if given as a parameter
 *         * @param threads will be filled with number of threads, if given as a parameter
 *          * @param blocks will be filled with number of blocks, if given as a parameter
 *           */
void parse_cmdline(
    const int argc,
    char**argv,
    size_t *heapInMB,
    unsigned *threads,
    unsigned *blocks,
    bool *machine_readable
    ){

  std::vector<std::pair<std::string, std::string> > parameters;

  // Parse Commandline, tokens are shaped like ARG=PARAM or ARG
  // This requires to use '=', if you want to supply a value with a parameter
  for (int i = 1; i < argc; ++i) {
    char* pos = strtok(argv[i], "=");
    std::pair < std::string, std::string > p(std::string(pos), std::string(""));
    pos = strtok(NULL, "=");
    if (pos != NULL) {
      p.second = std::string(pos);
    }
    parameters.push_back(p);
  }

  // go through all parameters that were found
  for (unsigned i = 0; i < parameters.size(); ++i) {
    std::pair < std::string, std::string > p = parameters.at(i);

    if (p.first == "-v" || p.first == "--verbose") {
      verbose = true;
    }

    if (p.first == "--threads") {
      *threads = atoi(p.second.c_str());
    }   

    if (p.first == "--blocks") {
      *blocks = atoi(p.second.c_str());
    }   

    if(p.first == "--heapsize") {
      *heapInMB = size_t(atoi(p.second.c_str()));
    }   

    if(p.first == "-h" || p.first == "--help"){
      print_help(argv);
      exit(0);
    }   

    if(p.first == "-m" || p.first == "--machine_readable"){
      *machine_readable = true;
    }   
  }
}

