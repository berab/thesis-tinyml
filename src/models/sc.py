import torch 
import torch.nn as nn


# Recurrent neural network (many-to-one)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, n_class, device, quant=None):
        super(RNN, self).__init__()
        self.quant = quant
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, n_class)
        self.device = device

    def forward(self, x):
        x = x.squeeze()
        x = x.transpose(1,2)
        # Set initial hidden and cell states
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device) 
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device) 
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # shape = (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out
 

class VGG(nn.Module):
    def __init__(self, ver, n_class, n_chan, quant=None, n_bits=None):
        super(VGG, self).__init__()
        config = cfg[ver]
        self.quant = quant
        self.n_chan = n_chan

        if quant == 'nvidia':
            import pytorch_quantization.nn as quant_nn
            from pytorch_quantization.tensor_quant import QuantDescriptor

            quant_desc_input = QuantDescriptor(num_bits=n_bits.x, fake_quant=True, axis=None, unsigned=False)
            quant_desc_weight = QuantDescriptor(num_bits=n_bits.W, fake_quant=True, axis=(0), unsigned=False)
            self.features = make_nvidia_layers(config, batch_norm=True, n_chan=n_chan,
                                    quant_desc_input=quant_desc_input, 
                                    quant_desc_weight=quant_desc_weight)
            self.classifier = nn.Sequential(
                quant_nn.Linear(512 * 1 * 1, 4096,
                                quant_desc_input=quant_desc_input, quant_desc_weight=quant_desc_weight),
                nn.ReLU(True),
                nn.Dropout(),
                quant_nn.Linear(4096, 4096,
                                quant_desc_input=quant_desc_input, quant_desc_weight=quant_desc_weight),
                nn.ReLU(True),
                nn.Dropout(),
                quant_nn.Linear(4096, n_class,
                                quant_desc_input=quant_desc_input, quant_desc_weight=quant_desc_weight),
            )

        elif quant == 'brevitas':
            import brevitas.nn as qnn
            self.features = make_brevitas_layers(config, batch_norm=True, n_chan=n_chan, n_bits=n_bits)
            self.classifier = nn.Sequential(
                qnn.QuantLinear(512 * 1 * 1, 4096, bias=True,
                                weight_bit_width=n_bits.W, return_quant_tensor=True),
                qnn.QuantReLU(inplace=True, bit_width=n_bits.x, return_quant_tensor=True),
                qnn.QuantDropout(return_quant_tensor=True),
                qnn.QuantLinear(4096, 4096, bias=True,
                                weight_bit_width=n_bits.W, return_quant_tensor=True),
                qnn.QuantReLU(inplace=True, bit_width=n_bits.x, return_quant_tensor=True),
                qnn.QuantDropout(return_quant_tensor=True),
                qnn.QuantLinear(4096, n_class, bias=True,
                                weight_bit_width=n_bits.W, return_quant_tensor=True),
            )

        else:
            self.features = make_layers(config, batch_norm=True, n_chan=n_chan)
            self.classifier = nn.Sequential(
                nn.Linear(512 * 1 * 1, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, n_class),
            )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512,
          512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512,
          'M', 512, 512, 512, 512, 'M'],}

def make_layers(cfg, batch_norm, n_chan):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(n_chan, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            n_chan = v
    return nn.Sequential(*layers)

def make_nvidia_layers(cfg, batch_norm, n_chan, quant_desc_input, quant_desc_weight):
    import pytorch_quantization.nn as quant_nn
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [quant_nn.MaxPool2d(kernel_size=2, stride=2, quant_desc_input=quant_desc_input)]
        else:
            conv2d = quant_nn.Conv2d(n_chan, v, kernel_size=3, padding=1,
                                     quant_desc_input=quant_desc_input,
                                     quant_desc_weight=quant_desc_weight)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            n_chan = v
    return nn.Sequential(*layers)

def make_brevitas_layers(cfg, batch_norm, n_chan, n_bits):
    import brevitas.nn as qnn
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [qnn.QuantMaxPool2d(kernel_size=2, stride=2, return_quant_tensor=True)]
        else:
            conv2d = qnn.QuantConv2d(n_chan, v, kernel_size=3, padding=1,
                                     weight_bit_width=n_bits.W, return_quant_tensor=True)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), qnn.QuantReLU(inplace=True, 
                           bit_width=n_bits.x, return_quant_tensor=True)]
            else:
                layers += [conv2d, qnn.QuantReLU(inplace=True, bit_width=n_bits.x,
                           return_quant_tensor=True)]
            n_chan = v
    return nn.Sequential(*layers)




# --- Test ---
def test_rnn():
    x = torch.ones(4,1,12,32)
    net = RNN(input_size=12, hidden_size=64, num_layers=3, n_class=35, device='cpu')
    output = net(x)
    n_params =  sum(p.numel() for p in net.parameters() if p.requires_grad)
    assert output.shape == (4, 35)
    assert n_params == 88803

def test_vgg19():
    x = torch.ones(4,1,32,32)
    ver = 'E'
    net = VGG(ver, n_class=35, n_chan=1)
    output = net(x)
    n_params =  sum(p.numel() for p in net.parameters() if p.requires_grad)

    assert output.shape == (4, 35)
    assert n_params == 39060195
