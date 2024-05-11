def save_checkpoint(state, is_best, checkpoint_dir='../checkpoints'):
    """
    checkpoint_dir is used to save the best checkpoint if this checkpoint is best one so far
    """
    checkpoint_path = os.path.join(checkpoint_dir,
                                   f"checkpoint_epoch{state['epoch']}_"
                                   f"{strftime('%Y-%m-%d-%H-%M-%S', localtime())}.pth.tar")
    torch.save(state, checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, os.path.join(checkpoint_dir, 'model_best.pth.tar'))
        
def model_summary(model):
  print("model_summary")
  print()
  print("Layer_name"+"\t"*7+"Number of Parameters")
  print("="*100)
  model_parameters = [layer for layer in model.parameters() if layer.requires_grad]
  layer_name = [child for child in model.children()]
  j = 0
  total_params = 0
  print("\t"*10)
  for i in layer_name:
    print()
    param = 0
    try:
        bias = (i.bias is not None)
    except:
        bias = False  
    if not bias:
        param =model_parameters[j].numel()+model_parameters[j+1].numel()
        j = j+2
    else:
        param =model_parameters[j].numel()
        j = j+1
    print(str(i)+"\t"*3+str(param))
    total_params+=param
  print("="*100)
  print(f"Total Params:{total_params}")       

def freeze_model_param(model):
    for i in [0, 3]:
        model.encoder1[i].weight.requires_grad = False 
        model.encoder2[i].weight.requires_grad = False
        model.encoder3[i].weight.requires_grad = False
        model.encoder4[i].weight.requires_grad = False

        model.bottleneck[i].weight.requires_grad = False

        model.decoder4[i].weight.requires_grad = False
        model.decoder3[i].weight.requires_grad = False
        model.decoder2[i].weight.requires_grad = False
        model.decoder1[i].weight.requires_grad = False
    
    for i in [1, 4]:
        model.encoder1[i].weight.requires_grad = False 
        model.encoder1[i].bias.requires_grad = False 

        model.encoder2[i].weight.requires_grad = False
        model.encoder2[i].bias.requires_grad = False

        model.encoder3[i].weight.requires_grad = False
        model.encoder3[i].bias.requires_grad = False

        model.encoder4[i].weight.requires_grad = False
        model.encoder4[i].bias.requires_grad = False

        model.bottleneck[i].weight.requires_grad = False
        model.bottleneck[i].bias.requires_grad = False

        model.decoder4[i].weight.requires_grad = False
        model.decoder4[i].bias.requires_grad = False

        model.decoder3[i].weight.requires_grad = False
        model.decoder3[i].bias.requires_grad = False

        model.decoder2[i].weight.requires_grad = False
        model.decoder2[i].bias.requires_grad = False

        model.decoder1[i].weight.requires_grad = False
        model.decoder1[i].bias.requires_grad = False


    model.upconv4.weight.requires_grad = False
    model.upconv4.bias.requires_grad = False

    model.upconv3.weight.requires_grad = False
    model.upconv3.bias.requires_grad = False

    model.upconv2.weight.requires_grad = False
    model.upconv2.bias.requires_grad = False

    model.upconv1.weight.requires_grad = False
    model.upconv1.bias.requires_grad = False

    model.conv_s.weight.requires_grad = False
    model.conv_s.bias.requires_grad = False

    return model

def print_network(model):
    print('model summary')
    for name, p in model.named_parameters():
        print(name)
        print(p.requires_grad)

def reinitialize_Siamese(model):
    torch.nn.init.xavier_uniform_(model.upconv4_c.weight)
    torch.nn.init.xavier_uniform_(model.upconv3_c.weight)
    torch.nn.init.xavier_uniform_(model.upconv2_c.weight)
    torch.nn.init.xavier_uniform_(model.upconv1_c.weight)
    torch.nn.init.xavier_uniform_(model.conv_c.weight)

    model.upconv4_c.bias.data.fill_(0.01)
    model.upconv3_c.bias.data.fill_(0.01)
    model.upconv2_c.bias.data.fill_(0.01)
    model.upconv1_c.bias.data.fill_(0.01)
    model.conv_c.bias.data.fill_(0.01)

    model.conv4_c.apply(init_weights)
    model.conv3_c.apply(init_weights)
    model.conv2_c.apply(init_weights)
    model.conv1_c.apply(init_weights)

    return model

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)