import plotly.graph_objs as go
import pickle
import numpy as np

# To use this just do
# import pickle
# from anim3d import Plotty_animation as pa
# f = pickle.load(open('appl_d_animd2o', 'rb'))
# pa(f)


# neural_network_outputs = np.random.randn(generations, individual_data_outputs, output_neurons)

class Plotty_animation:
    # Takes in 3D array as argument. Can be: first dimension as generations 2nd as outputs and 3rd as dataset
    def __init__(self, argument) -> None:
        neural_network_outputs = argument
        generations = neural_network_outputs.shape[0]
        individual_generation_output = neural_network_outputs.shape[1]
        output_neurons = neural_network_outputs.shape[2]

        frames = []
        for each_gen in range(generations):
            frame = {'data': [], 'name': f'Frame {each_gen}'}
            for each_neuron in range(output_neurons):
                scatter = go.Scatter(
                    x=np.arange(individual_generation_output),
                    y=neural_network_outputs[each_gen, :, each_neuron],
                    mode='markers',
                    name=f'Neuron {each_neuron}'
                )
                frame['data'].append(scatter)
            frames.append(frame)

        # initial figure it seems
        fig = go.Figure(
            data=frames[0]['data'],
            layout=go.Layout(
                xaxis=dict(title='Data Point'),
                yaxis=dict(range=[np.min(neural_network_outputs), np.max(neural_network_outputs)], title='Output Values'),
                title='Neural Network Output Evolution'
            ),
            frames=frames
        )

        fig.update_layout(updatemenus=[{
            'buttons': [
                {
                    'args': [None, {'frame': {'duration': 50, 'redraw': True}, 'fromcurrent': True}],
                    'label': 'Play',
                    'method': 'animate',
                },
                {
                    'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 0}}],
                    'label': 'Pause',
                    'method': 'animate',
                },
            ],
            'direction': 'left',
            'showactive': False,
            'type': 'buttons',
            'x': 0,
            'y':-0.1
        }], sliders=[{
                    'active': 0,
                    'currentvalue': {'prefix': 'Gen: '},
                    'steps': [{
                        'args': [[f'Frame {i}'], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}],
                        'label': f'{i}',
                        'method': 'animate'
                    } for i in range(len(frames))],
                    'x': 0.1,  # Adjust the x value to position the slider
                    'y': -0.1,  # Adjust the y value to position the slider at the bottom
                }])

        fig.show()
