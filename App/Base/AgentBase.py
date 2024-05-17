
class AgentBase():
    def __init__(self):
        pass

    def take_action(self,state,*args,**kwargs):
        """
        return Action According to given state
        :param state:
        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    def update(self,transition_dict,*args,**kwargs):
        """
        Update Params According to given transition_dict to Modify your Agent
        :param transition_dict:
        transition_dict = {
            'states': state_i,
            'actions': action_i,
            'next_states': n_state_i,
            'rewards': reward_i,
            'dones': done_i
        }
        """