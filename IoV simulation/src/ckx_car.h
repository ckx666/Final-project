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

#ifndef __CKX_CKX_CAR_H_
#define __CKX_CKX_CAR_H_

#include <omnetpp.h>
#include <inet/transportlayer/contract/udp/UdpSocket.h>
#include <inet/networklayer/common/L3Address.h>
#include <inet/networklayer/common/L3AddressResolver.h>
#include <inet/mobility/contract/IMobility.h>
#include <veins/modules/mobility/traci/TraCICommandInterface.h>
#include "veins_inet/VeinsInetMobility.h"
#include <set>
#include <random>
#include "aoid_m.h"
#include "dl_m.h"
using namespace omnetpp;
using namespace inet;

std::set<int> allNodeIndex;
class Ckx_car : public cSimpleModule
{
  protected:
    virtual int numInitStages() const override { return NUM_INIT_STAGES; }
    void initialize(int stage) override;
    virtual void handleMessage(cMessage *msg) override;
    virtual void finish() override{}
};

#endif
