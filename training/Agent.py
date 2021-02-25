import numpy as np

class Agent:

    def __init__(self, start, size, osize):
        self.yaw = 90
        self.position = np.array(start)
        self.shiftArr = {180:(0,-1), 270:(1,0), 0: (0,1), 90: (-1,0)}
        self.size = size
        self.framing = size//2 + osize



    def reset(self, start):
        self.yaw = 90
        self.position = np.array(start)

    def getPosition(self):
        return self.position

    def getYaw(self):
        return self.yaw

    def doAction(self, command, array):
        if command == "move 1":
            if self._diffForFacingBlock(array) <= 0:
                self.position += self.shiftArr[self.yaw]
        elif command == "turn 1":
            self.yaw +=90
            self.yaw %=360
        elif command == "turn -1":
            self.yaw -= 90
            self.yaw %= 360
        elif command == "jumpmove 1":
            if self._diffForFacingBlock(array) <= 1:
                self.position += self.shiftArr[self.yaw]
        else:
            print("ERROR: COMMAND DOES NOT EXIST")

    def _diffForFacingBlock(self, array):
        curr = array[self.position[1] +self.framing, self.position[0] + self.framing]
        look = array[self.position[1] + self.shiftArr[self.yaw][1] +self.framing, self.position[0] + self.shiftArr[self.yaw][0] + self.framing]
        #print("[AGENT] diffBlock:",look-curr)
        return look-curr


