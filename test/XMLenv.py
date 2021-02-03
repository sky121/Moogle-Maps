from builtins import range
import time
from PerlinNoiseFactory import PerlinNoiseFactory
import numpy as np
import math
import random

class XMLenv:
    def __init__(self, size = 201):
        self.size = size
        self.terrain_array = self.getTerrain()
        self.center = self.size//2
        self.obs_size = 5
        self.max_episode_steps = 100
        i = math.floor(random.random()*size)
        j = math.floor(random.random()*size)

        #coordinate in the form of (x, y, z)
        self.end_coordinate = (i-self.center,self.terrain_array[i,j]+1,j-self.center)  #FLOOR AGENTS X AND Z COORDINATES TO CHECK IF ITS AT THE END COORDINATE
        self.start_coordinate = (0.5, self.terrain_array[self.size//2,self.size//2]+1, 0.5) 

    def getTerrain(self):
        p = PerlinNoiseFactory(2,4)
        a = np.array([[p(i/self.size,j/self.size) for j in range(self.size)] for i in range(self.size)])
        a = np.abs((a*50).astype(int)) + 5
        return a

    def Menger(self, blocktype, walltype):

        #draw solid chunk
        genstring = ""
        #now remove holes
  
        
        for i in range(self.size):
            for j in range(self.size):
                #clear the old stones first since malmo does not rebuild and clear for us
                genstring += self.drawLine(i-self.center,0,j-self.center,i-self.center,100,j-self.center,"air")+ "\n"
                genstring += self.drawLine(i-self.center,0,j-self.center,i-self.center,self.terrain_array[i,j],j-self.center,blocktype)+ "\n"


        L = -self.center-1
        h = self.size-self.center
        for sx,sz,ex,ez in ((h,L,L,L),(h,L,h,h),(h,h,L,h),(L,L,L,h)):
            genstring += self.drawCuboid(sx,0,sz,ex,50,ez,walltype) + "\n"
        
        return genstring

    def drawLine(self, x1, y1, z1, x2, y2, z2, blocktype):

        if x2 == math.floor(self.start_coordinate[0]) and y2 == self.start_coordinate[1]-1 and z2 == math.floor(self.start_coordinate[2]):
            blocktype = "diamond_block"
           
        if x2 == self.end_coordinate[0] and y2 == self.end_coordinate[1]-1 and z2 == self.end_coordinate[2]:
            print("END COORDINATE:",self.end_coordinate)
            blocktype = "red_sandstone"
        return '<DrawLine x1="' + str(x1) + '" y1="' + str(y1) + '" z1="' + str(z1) + '" x2="' + str(x2) + '" y2="' + str(y2) + '" z2="' + str(z2) + '" type="' + blocktype + '"/>'

    def drawCuboid(self, x1, y1, z1, x2, y2, z2, blocktype):
        return '<DrawCuboid x1="' + str(x1) + '" y1="' + str(y1) + '" z1="' + str(z1) + '" x2="' + str(x2) + '" y2="' + str(y2) + '" z2="' + str(z2) + '" type="' + blocktype + '"/>'

    def generateWorldXML(self, blocktype):
        missionXML='''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
            <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
            
              <About>
                <Summary>Hello world!</Summary>
              </About>
              
            <ServerSection>
              <ServerInitialConditions>
                <Time>
                    <StartTime>1000</StartTime>
                    <AllowPassageOfTime>false</AllowPassageOfTime>
                </Time>
                <Weather>clear</Weather>
              </ServerInitialConditions>
              <ServerHandlers>
                  <FlatWorldGenerator generatorString=";"/>
                  <DrawingDecorator>
                    ''' + self.Menger(blocktype,"glass") + '''
                  </DrawingDecorator>
                  <ServerQuitFromTimeUp timeLimitMs="30000"/>
                  <ServerQuitWhenAnyAgentFinishes/>
                </ServerHandlers>
              </ServerSection>
              
              <AgentSection mode="Survival">
                <Name>MalmoTutorialBot</Name>
                <AgentStart>
                    <Placement x="'''+str(self.start_coordinate[0])+'''" y="'''+str(self.start_coordinate[1])+'''" z="'''+str(self.start_coordinate[2])+'''" yaw="90"/>
      
                </AgentStart>
                <AgentHandlers>
                  <DiscreteMovementCommands/>
                  <ObservationFromFullStats/>
                  <ObservationFromRay/>
                  <ObservationFromGrid>
                      <Grid name="floorAll">
                        <min x="-'''+str(int(self.obs_size/2))+'''" y="-1" z="-'''+str(int(self.obs_size/2))+'''"/>
                        <max x="'''+str(int(self.obs_size/2))+'''" y="0" z="'''+str(int(self.obs_size/2))+'''"/>
                      </Grid>
                  </ObservationFromGrid>
                  
                  <InventoryCommands/>
                  <AgentQuitFromReachingCommandQuota total="'''+str(self.max_episode_steps*3)+'''" />
                  <AgentQuitFromTouchingBlockType>
                      <Block type="red_sandstone" />
                  </AgentQuitFromTouchingBlockType>
                </AgentHandlers>
              </AgentSection>
            </Mission>'''
        return missionXML