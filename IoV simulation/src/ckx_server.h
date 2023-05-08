//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
// 
// You should have received a copy of the GNU Lesser General Public License
// along with this program.  If not, see http://www.gnu.org/licenses/.
// 

#ifndef __CKX_CKX_SERVER_H_
#define __CKX_CKX_SERVER_H_

#include <omnetpp.h>
#include <inet/transportlayer/contract/udp/UdpSocket.h>
#include <inet/networklayer/common/L3Address.h>
#include <inet/networklayer/common/L3AddressResolver.h>
#include <inet/mobility/contract/IMobility.h>
#include <veins/modules/mobility/traci/TraCICommandInterface.h>
#include "veins_inet/VeinsInetMobility.h"
#include <inet/common/ModuleAccess.h>
#include <string>
#include <set>
#include <vector>
#include <unordered_map>
#include <random>
#include <algorithm>
#include <fstream>

#include "dl_m.h"
#include "aoid_m.h"

using namespace omnetpp;
using namespace inet;

extern std::set<int> allNodeIndex;

class Ckx_server : public cSimpleModule{
private:
    cMessage *selfMsg = nullptr;
    cMessage *triggerMsg = nullptr;
    int total_car_number;
    double period_;
    std::vector<std::vector<int>> AoI_data;
    int count;

    std::unordered_map<int, std::vector<double>> weight_probility;
    std::unordered_map<int, std::vector<double>> success_probility;
  protected:
    virtual int numInitStages() const override { return NUM_INIT_STAGES; }
    void initialize(int stage) override;
    void handleMessage(cMessage *msg) override;
    void finish(){
        double sum_fin = 0.;
        for(auto ea : AoI_data){
            int tmp = std::accumulate(begin(ea), end(ea), 0);
            sum_fin += tmp / ea.size();
        }
        std::ofstream fd;
        fd.open(std::string("./ckx_aoi_result2/") + std::to_string(total_car_number) + ".txt", std::ios::out | std::ios::app);
        fd << sum_fin / total_car_number << "\n";
        fd.close();
    }
  public:
    int select_node_upload();
    void update_AoI(int node_index);
    void update_weight_probility();
    int select_node_upload1();
};

#endif
