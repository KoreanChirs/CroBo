def observation(self, observation):
        ### INPUT SHOULD BE [0,255]
        if self.embedding is None:
            return observation

        if "vc1" in self.load_path or "vc_1" in self.load_path:
            # transforms가 이미 (np.ndarray -> tensor[1,3,224,224]) 람다임
            inp = self.transforms(observation.astype(np.uint8))
        else:
            inp = self.transforms(Image.fromarray(observation.astype(np.uint8))).reshape(-1, 3, 224, 224)

        if "r3m" in self.load_path:
            # R3M expects 0-255; ToTensor gave 0-1
            inp = inp * 255.0

        inp = inp.to(self.device)

        with torch.no_grad():
            emb_t = self.embedding(inp)   # VC1Enc는 (B,D) 보장
            if torch.is_tensor(emb_t):
                emb = emb_t.view(-1, self.embedding_dim).to('cpu').numpy().squeeze()
            else:
                raise RuntimeError("Embedding output is not a tensor")

        if self.proprio:
            try:
                proprio = self.env.unwrapped.get_obs()[:self.proprio]
            except:
                proprio = self.env.unwrapped._get_obs()[:self.proprio]
            emb = np.concatenate([emb, proprio])

        return emb


    def encode_batch(self, obs, finetune=False):
        ### INPUT SHOULD BE [0,255]
        inp_list = []

        for o in obs:
            o = o.astype(np.uint8)
            if "vc1" in self.load_path or "vc_1" in self.load_path:
                i = self.transforms(o)  # already [1,3,224,224]
            else:
                i = self.transforms(Image.fromarray(o)).reshape(-1, 3, 224, 224)

            if "r3m" in self.load_path:
                i = i * 255.0
            inp_list.append(i)

        inp = torch.cat(inp_list, dim=0).to(self.device)

        if finetune and self.start_finetune:
            emb = self.embedding(inp)  # keep tensor
            emb = emb.view(-1, self.embedding_dim)
        else:
            with torch.no_grad():
                emb = self.embedding(inp).view(-1, self.embedding_dim).to('cpu').numpy().squeeze()
        return emb