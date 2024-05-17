import gym

class CartPole():
    def __init__(self):
        env_name = 'CartPole-v1'
        self = gym.make(env_name)
        # self = env
        # self.__dict__.update(env.__dict__)

    # def __getattr__(self, item):
    #     if item in self.env.__dict__:
    #         return getattr(self.env,item)

if __name__ == '__main__':
    c = CartPole()
    print(c.action_space.n)
