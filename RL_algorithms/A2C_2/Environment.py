class Environment:
    def get_state(self, env, _obs):
        env_low = env.observation_space.low 
        env_high = env.observation_space.high 
        # normalize
        state = (_obs - env_low)/(env_high - env_low)
        return state.tolist()

    def get_reward(self, env, reward):
        reward = float(reward)
        #env_low = env.observation_space.low 
        #env_high = env.observation_space.high 
        # normalize
        #state = (_obs - env_low)/(env_high - env_low)
        return reward
    def get_action(self,env, action):
        action = int(action)

        return action

    def get_data(self, env, obs, next_obs, reward, action):
        state = self.get_state(env, obs)
        next_state = self.get_state(env, next_obs)
        reward = self.get_reward(env, reward)
        action = self.get_action(env, action)

        return state, action, next_state, reward