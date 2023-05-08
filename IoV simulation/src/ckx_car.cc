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

#include "ckx_car.h"

Define_Module(Ckx_car);
void Ckx_car::initialize(int stage){
    cSimpleModule::initialize(stage);
    if (stage != INITSTAGE_APPLICATION_LAYER) return;

    allNodeIndex.insert(getParentModule()->getIndex());
}

void Ckx_car::handleMessage(cMessage *msg){
    DL* mmsg = check_and_cast<DL*>(msg);
    double pro = mmsg->getSuccess_probility();
    delete mmsg;
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<double> distribution_probility(0, 1);
    if(distribution_probility(generator) >= pro) return;
    AoId* packet = new AoId("upload");
    packet->setWhich(getParentModule()->getIndex());
    cModule *pp_sub = getParentModule()->getParentModule()->getSubmodule("server");
    if(pp_sub != nullptr){
        sendDirect(packet, pp_sub->getSubmodule("app", 0)->gate("hostIn"));
    }else{
        printf("pp_sub is null pointer! \n");
    }
}
