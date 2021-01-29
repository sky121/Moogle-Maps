
from XMLenv import XMLenv
from malmo import MalmoPython
import json
import sys
import os
import time


def startMission(agent_host,my_mission, my_mission_record,max_retries):
    for retry in range(max_retries):
        try:
            agent_host.startMission( my_mission, my_mission_record )
            break
        except RuntimeError as e:
            if retry == max_retries - 1:
                print("Error starting mission:",e)
                exit(1)
            else:
                time.sleep(2)

def main():
    agent_host = MalmoPython.AgentHost()
    try:
        agent_host.parse( sys.argv )
    except RuntimeError as e:
        print('ERROR:',e)
        print(agent_host.getUsage())
        exit(1)
    if agent_host.receivedArgument("help"):
        print(agent_host.getUsage())
        exit(0)
    environment = XMLenv()
    my_mission = MalmoPython.MissionSpec(environment.generateWorldXML("stone"), True)
    my_mission_record = MalmoPython.MissionRecordSpec()

    # Attempt to start a mission:
    startMission(agent_host,my_mission, my_mission_record,3)

    # Loop until mission starts:
    print("Waiting for the mission to start ", end=' ')
    world_state = agent_host.getWorldState()
    while not world_state.has_mission_begun:
        print(".", end="")
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:",error.text)

    print()
    print("Mission running ", end=' ')

    # Loop until mission ends:
    while world_state.is_mission_running:
        print(".", end="")
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:",error.text)
        if world_state.number_of_observations_since_last_state > 0: # Have any observations come in?
            msg = world_state.observations[-1].text                 # Yes, so get the text
            observations = json.loads(msg)                          # and parse the JSON
            grid = observations.get(u'floor3x3', 0)                 # and get the grid we asked for
            # ADD SOME CODE HERE TO SAVE YOUR AGENT
    print()
    print("Mission ended")


if __name__ == "__main__":
    main()