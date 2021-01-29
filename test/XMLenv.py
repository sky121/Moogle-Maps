from builtins import range
import time
from PerlinNoiseFactory import PerlinNoiseFactory
import numpy as np

class XMLenv:
    def __init__(self, size = 201):
        self.size = size
        self.terrain_array = self.getTerrain()

    def getTerrain(self):
        p = PerlinNoiseFactory(2,4)
        a = np.array([[p(i/self.size,j/self.size) for j in range(self.size)] for i in range(self.size)])
        a = np.abs((a*50).astype(int))
        return a

    def Menger(self, blocktype, walltype):

        #draw solid chunk
        genstring = ""
        #now remove holes
        center = self.size//2
        
        for i in range(self.size):
            for j in range(self.size):
                genstring += self.drawLine(i-center,0,j-center,i-center,self.terrain_array[i,j]+5,j-center,blocktype)+ "\n"


        L = -center-1
        h = self.size-center
        for sx,sz,ex,ez in ((h,L,L,L),(h,L,h,h),(h,h,L,h),(L,L,L,h)):
            genstring += self.drawCuboid(sx,0,sz,ex,50,ez,walltype) + "\n"
        
        return genstring

    def drawLine(self, x1, y1, z1, x2, y2, z2, blocktype):
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
                    <Placement x="0.5" y="''' + str(self.terrain_array[self.size//2,self.size//2]+7) + '''" z="0.5" yaw="90"/>
                    <Inventory>
                        <InventoryItem slot="8" type="diamond_pickaxe"/>
                    </Inventory>
                </AgentStart>
                <AgentHandlers>
                  <ObservationFromFullStats/>
                  <ObservationFromGrid>
                      <Grid name="floor3x3">
                        <min x="-1" y="-1" z="-1"/>
                        <max x="1" y="-1" z="1"/>
                      </Grid>
                  </ObservationFromGrid>
                  <ContinuousMovementCommands turnSpeedDegs="180"/>
                  <InventoryCommands/>
                  <AgentQuitFromTouchingBlockType>
                      <Block type="diamond_block" />
                  </AgentQuitFromTouchingBlockType>
                </AgentHandlers>
              </AgentSection>
            </Mission>'''
        return missionXML