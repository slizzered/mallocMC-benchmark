//basic files for mallocMC
#include <mallocMC/mallocMC_hostclass.hpp>

// Load all available policies for mallocMC
#include <mallocMC/CreationPolicies.hpp>
#include <mallocMC/DistributionPolicies.hpp>
#include <mallocMC/OOMPolicies.hpp>
#include <mallocMC/ReservePoolPolicies.hpp>
#include <mallocMC/AlignmentPolicies.hpp>

#include <boost/mpl/int.hpp>
#include <boost/mpl/bool.hpp>


//configurate the CreationPolicy "Scatter"
struct ScatterConfig{
  typedef boost::mpl::int_<4096>  pagesize;
  typedef boost::mpl::int_<8>     accessblocks;
  typedef boost::mpl::int_<16>    regionsize;
  typedef boost::mpl::int_<2>     wastefactor;
  typedef boost::mpl::bool_<false> resetfreedpages;
};

struct ScatterHashParams{
  typedef boost::mpl::int_<38183> hashingK;
  typedef boost::mpl::int_<17497> hashingDistMP;
  typedef boost::mpl::int_<1>     hashingDistWP;
  typedef boost::mpl::int_<1>     hashingDistWPRel;
};

// configure the DistributionPolicy "XMallocSIMD"
struct DistributionConfig{
  typedef ScatterConfig::pagesize pagesize;
};

// configure the AlignmentPolicy "Shrink"
struct AlignmentConfig{
  typedef boost::mpl::int_<16> dataAlignment;
};

// Define a new allocator and call it ScatterAllocator
// which resembles the behaviour of ScatterAlloc
typedef mallocMC::Allocator<
mallocMC::CreationPolicies::Scatter<ScatterConfig,ScatterHashParams>,
  mallocMC::DistributionPolicies::XMallocSIMD<DistributionConfig>,
  mallocMC::OOMPolicies::ReturnNull,
  mallocMC::ReservePoolPolicies::SimpleCudaMalloc,
  mallocMC::AlignmentPolicies::Shrink<AlignmentConfig>
  > ScatterAllocator;

