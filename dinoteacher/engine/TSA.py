class TeacherStudentAlignHead(nn.Module):
    def __init__(self, cfg, student_dim, teacher_dim, normalize_feature=True):
        super(TeacherStudentAlignHead, self).__init__()
        head_type = cfg.SEMISUPNET.ALIGN_HEAD_TYPE
        self.proj_dim = cfg.SEMISUPNET.ALIGN_HEAD_PROJ_DIM
        self.normalize_feature = normalize_feature
        if cfg.SEMISUPNET.ALIGN_PROJ_GELU:
            nl_layer = nn.GELU()
        else:
            nl_layer = nn.ReLU()
        if head_type=='attention':
            self.projection_layer = MHALayer(student_dim, teacher_dim)
        elif head_type=='MLP':
            self.projection_layer = nn.Sequential(nn.Conv2d(student_dim, self.proj_dim, 1, 1),
                                                   nl_layer,
                                                   nn.Conv2d(self.proj_dim, teacher_dim, 1, 1))
        elif head_type=='MLP3':
            self.projection_layer = nn.Sequential(nn.Conv2d(student_dim, self.proj_dim, 1, 1),
                                                   nl_layer,
                                                   nn.Conv2d(self.proj_dim, self.proj_dim, 1, 1),
                                                   nl_layer,
                                                   nn.Conv2d(self.proj_dim, teacher_dim, 1, 1))
        elif head_type=='linear':
            self.projection_layer = nn.Conv2d(student_dim, teacher_dim, 1, 1)
        elif head_type=='Direct':
            self.projection_layer = nn.Identity()
        else:
            raise NotImplementedError("{} align head not supported.".format(head_type))
        

    def forward(self, feat_cnn, teacher_feat_shape):
        return self.project_student_feat(feat_cnn, teacher_feat_shape)
    
    def project_student_feat(self, feat_cnn, teacher_feat_shape):
        h, w = teacher_feat_shape
        feat_cnn = self.projection_layer(feat_cnn)
        feat_cnn = F.interpolate(feat_cnn, (h,w), mode='bilinear')
        if self.normalize_feature:
            feat_cnn = F.normalize(feat_cnn, p=2, dim=1)
        return feat_cnn
            
    def align_loss(self, feat_student, feat_teacher, return_sim=False):
        if self.normalize_feature:
            
            
            loss = F.l1_loss(feat_student, feat_teacher, reduction='mean')
            
            
            
            
            

        else:
            sim = torch.linalg.norm(feat_student-feat_teacher, dim=1, ord=2)
            loss = sim.mean() / 100

        if return_sim:
            return loss, sim
        else:
            return loss