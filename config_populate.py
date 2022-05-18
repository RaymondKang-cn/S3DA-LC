data_settings = {'office-31': {},
                 'office-caltech': {},
                 'office-home': {},
                 'domain-net': {},
                 'image-clef': {}}

data_settings['office-31']['DW_A'] = {}
data_settings['office-31']['AD_W'] = {}
data_settings['office-31']['AW_D'] = {}

data_settings['office-home']['ACP_R'] = {}
data_settings['office-home']['ACR_P'] = {}
data_settings['office-home']['APR_C'] = {}
data_settings['office-home']['CPR_A'] = {}

data_settings['office-caltech']['ACD_W'] = {}
data_settings['office-caltech']['ADW_C'] = {}
data_settings['office-caltech']['ACW_D'] = {}
data_settings['office-caltech']['CDW_A'] = {}

data_settings['domain-net']['CIPQR_S'] = {}
data_settings['domain-net']['CIPQS_R'] = {}
data_settings['domain-net']['CIPSR_Q'] = {}
data_settings['domain-net']['CPQRS_I'] = {}
data_settings['domain-net']['CIQRS_P'] = {}
data_settings['domain-net']['IPQRS_C'] = {}

data_settings['image-clef']['PC_I'] = {}
data_settings['image-clef']['IC_P'] = {}
data_settings['image-clef']['IP_C'] = {}

data_settings['office-31']['DW_A']['C'] = {
    'dslr': ['desk_chair', 'monitor', 'printer', 'keyboard', 'ring_binder', 'speaker', 'tape_dispenser', 'bike',
             'headphones', 'mobile_phone', 'desk_lamp', 'ruler', 'calculator', 'desktop_computer', 'trash_can', 'mouse',
             'laptop_computer', 'mug', 'punchers', 'phone', 'back_pack', 'bike_helmet', 'file_cabinet',
             'paper_notebook', 'letter_tray', 'bookcase', 'pen', 'scissors', 'projector', 'bottle', 'stapler'],
    'webcam': ['desk_chair', 'monitor', 'printer', 'keyboard', 'ring_binder', 'speaker', 'tape_dispenser', 'bike',
               'headphones', 'mobile_phone', 'desk_lamp', 'ruler', 'calculator', 'desktop_computer', 'trash_can',
               'mouse', 'laptop_computer', 'mug', 'punchers', 'phone', 'back_pack', 'bike_helmet', 'file_cabinet',
               'paper_notebook', 'letter_tray', 'bookcase', 'pen', 'scissors', 'projector', 'bottle', 'stapler']
}

data_settings['office-31']['AW_D']['C'] = {
    'amazon': ['desk_chair', 'monitor', 'printer', 'keyboard', 'ring_binder', 'speaker', 'tape_dispenser', 'bike',
               'headphones', 'mobile_phone', 'desk_lamp', 'ruler', 'calculator', 'desktop_computer', 'trash_can',
               'mouse', 'laptop_computer', 'mug', 'punchers', 'phone', 'back_pack', 'bike_helmet', 'file_cabinet',
               'paper_notebook', 'letter_tray', 'bookcase', 'pen', 'scissors', 'projector', 'bottle', 'stapler'],
    'webcam': ['desk_chair', 'monitor', 'printer', 'keyboard', 'ring_binder', 'speaker', 'tape_dispenser', 'bike',
               'headphones', 'mobile_phone', 'desk_lamp', 'ruler', 'calculator', 'desktop_computer', 'trash_can',
               'mouse', 'laptop_computer', 'mug', 'punchers', 'phone', 'back_pack', 'bike_helmet', 'file_cabinet',
               'paper_notebook', 'letter_tray', 'bookcase', 'pen', 'scissors', 'projector', 'bottle', 'stapler']
}

data_settings['office-31']['AD_W']['C'] = {
    'amazon': ['desk_chair', 'monitor', 'printer', 'keyboard', 'ring_binder', 'speaker', 'tape_dispenser', 'bike',
               'headphones', 'mobile_phone', 'desk_lamp', 'ruler', 'calculator', 'desktop_computer', 'trash_can',
               'mouse', 'laptop_computer', 'mug', 'punchers', 'phone', 'back_pack', 'bike_helmet', 'file_cabinet',
               'paper_notebook', 'letter_tray', 'bookcase', 'pen', 'scissors', 'projector', 'bottle', 'stapler'],
    'dslr': ['desk_chair', 'monitor', 'printer', 'keyboard', 'ring_binder', 'speaker', 'tape_dispenser', 'bike',
             'headphones', 'mobile_phone', 'desk_lamp', 'ruler', 'calculator', 'desktop_computer', 'trash_can', 'mouse',
             'laptop_computer', 'mug', 'punchers', 'phone', 'back_pack', 'bike_helmet', 'file_cabinet',
             'paper_notebook', 'letter_tray', 'bookcase', 'pen', 'scissors', 'projector', 'bottle', 'stapler']
}

data_settings['office-31']['AW_D']['C_dash'] = {
    'amazon': [],
    'dslr': [],
    'webcam': []
}

data_settings['office-31']['DW_A']['C_dash'] = {
    'amazon': [],
    'dslr': [],
    'webcam': []
}

data_settings['office-31']['AD_W']['C_dash'] = {
    'amazon': [],
    'dslr': [],
    'webcam': []
}

data_settings['office-31']['DW_A']['src_datasets'] = ['dslr', 'webcam']
data_settings['office-31']['AW_D']['src_datasets'] = ['amazon', 'webcam']
data_settings['office-31']['AD_W']['src_datasets'] = ['amazon', 'dslr']

data_settings['office-31']['DW_A']['trgt_datasets'] = ['amazon']
data_settings['office-31']['AW_D']['trgt_datasets'] = ['dslr']
data_settings['office-31']['AD_W']['trgt_datasets'] = ['webcam']

data_settings['office-31']['DW_A']['num_C'] = {'dslr': len(data_settings['office-31']['DW_A']['C']['dslr']),
                                               'webcam': len(data_settings['office-31']['DW_A']['C']['webcam']),
                                               'amazon': len(set(data_settings['office-31']['DW_A']['C']['dslr'] +
                                                                 data_settings['office-31']['DW_A']['C']['webcam']))}

data_settings['office-31']['AW_D']['num_C'] = {'amazon': len(data_settings['office-31']['AW_D']['C']['amazon']),
                                               'webcam': len(data_settings['office-31']['AW_D']['C']['webcam']),
                                               'dslr': len(set(data_settings['office-31']['AW_D']['C']['amazon'] +
                                                               data_settings['office-31']['AW_D']['C']['webcam']))
                                               }

data_settings['office-31']['AD_W']['num_C'] = {'amazon': len(data_settings['office-31']['AD_W']['C']['amazon']),
                                               'dslr': len(data_settings['office-31']['AD_W']['C']['dslr']),
                                               'webcam': len(set(data_settings['office-31']['AD_W']['C']['amazon'] +
                                                                 data_settings['office-31']['AD_W']['C']['dslr']))}

data_settings['office-31']['DW_A']['num_C_dash'] = {'dslr': len(data_settings['office-31']['DW_A']['C_dash']['dslr']),
                                                    'webcam': len(
                                                        data_settings['office-31']['DW_A']['C_dash']['webcam']),
                                                    'amazon': len(
                                                        data_settings['office-31']['DW_A']['C_dash']['amazon'])}

data_settings['office-31']['AW_D']['num_C_dash'] = {'dslr': len(data_settings['office-31']['AW_D']['C_dash']['dslr']),
                                                    'webcam': len(
                                                        data_settings['office-31']['AW_D']['C_dash']['webcam']),
                                                    'amazon': len(
                                                        data_settings['office-31']['AW_D']['C_dash']['amazon'])}

data_settings['office-31']['AD_W']['num_C_dash'] = {'dslr': len(data_settings['office-31']['AD_W']['C_dash']['dslr']),
                                                    'webcam': len(
                                                        data_settings['office-31']['AD_W']['C_dash']['webcam']),
                                                    'amazon': len(
                                                        data_settings['office-31']['AD_W']['C_dash']['amazon'])}

data_settings['office-home']['ACP_R']['C'] = {
    'Art': ['Sneakers', 'Mouse', 'Flowers', 'Bottle', 'Fork', 'Fan', 'Refrigerator', 'Speaker', 'Calculator',
            'Clipboards', 'Marker', 'Batteries', 'Lamp_Shade', 'Hammer', 'Kettle', 'Mug', 'Ruler', 'Sink', 'Pen',
            'File_Cabinet', 'Backpack', 'Keyboard', 'Desk_Lamp', 'Glasses', 'Soda', 'Spoon', 'Mop', 'Computer', 'Couch',
            'ToothBrush', 'Toys', 'Knives', 'Drill', 'Calendar', 'Screwdriver', 'Candles', 'Curtains', 'Eraser', 'Pan',
            'Printer', 'Table', 'TV', 'Telephone', 'Push_Pin', 'Laptop', 'Chair', 'Flipflops', 'Scissors', 'Bed',
            'Paper_Clip', 'Monitor', 'Bike', 'Postit_Notes', 'Alarm_Clock', 'Webcam', 'Shelf', 'Exit_Sign', 'Notebook',
            'Oven', 'Trash_Can', 'Radio', 'Bucket', 'Folder', 'Pencil', 'Helmet'],
    'Clipart': ['Sneakers', 'Mouse', 'Flowers', 'Bottle', 'Fork', 'Fan', 'Refrigerator', 'Speaker', 'Calculator',
                'Clipboards', 'Marker', 'Batteries', 'Lamp_Shade', 'Hammer', 'Kettle', 'Mug', 'Ruler', 'Sink', 'Pen',
                'File_Cabinet', 'Backpack', 'Keyboard', 'Desk_Lamp', 'Glasses', 'Soda', 'Spoon', 'Mop', 'Computer',
                'Couch', 'ToothBrush', 'Toys', 'Knives', 'Drill', 'Calendar', 'Screwdriver', 'Candles', 'Curtains',
                'Eraser', 'Pan', 'Printer', 'Table', 'TV', 'Telephone', 'Push_Pin', 'Laptop', 'Chair', 'Flipflops',
                'Scissors', 'Bed', 'Paper_Clip', 'Monitor', 'Bike', 'Postit_Notes', 'Alarm_Clock', 'Webcam', 'Shelf',
                'Exit_Sign', 'Notebook', 'Oven', 'Trash_Can', 'Radio', 'Bucket', 'Folder', 'Pencil', 'Helmet'],
    'Product': ['Sneakers', 'Mouse', 'Flowers', 'Bottle', 'Fork', 'Fan', 'Refrigerator', 'Speaker', 'Calculator',
                'Clipboards', 'Marker', 'Batteries', 'Lamp_Shade', 'Hammer', 'Kettle', 'Mug', 'Ruler', 'Sink', 'Pen',
                'File_Cabinet', 'Backpack', 'Keyboard', 'Desk_Lamp', 'Glasses', 'Soda', 'Spoon', 'Mop', 'Computer',
                'Couch', 'ToothBrush', 'Toys', 'Knives', 'Drill', 'Calendar', 'Screwdriver', 'Candles', 'Curtains',
                'Eraser', 'Pan', 'Printer', 'Table', 'TV', 'Telephone', 'Push_Pin', 'Laptop', 'Chair', 'Flipflops',
                'Scissors', 'Bed', 'Paper_Clip', 'Monitor', 'Bike', 'Postit_Notes', 'Alarm_Clock', 'Webcam', 'Shelf',
                'Exit_Sign', 'Notebook', 'Oven', 'Trash_Can', 'Radio', 'Bucket', 'Folder', 'Pencil', 'Helmet']
}

data_settings['office-home']['ACR_P']['C'] = {
    'Art': ['Sneakers', 'Mouse', 'Flowers', 'Bottle', 'Fork', 'Fan', 'Refrigerator', 'Speaker', 'Calculator',
            'Clipboards', 'Marker', 'Batteries', 'Lamp_Shade', 'Hammer', 'Kettle', 'Mug', 'Ruler', 'Sink', 'Pen',
            'File_Cabinet', 'Backpack', 'Keyboard', 'Desk_Lamp', 'Glasses', 'Soda', 'Spoon', 'Mop', 'Computer', 'Couch',
            'ToothBrush', 'Toys', 'Knives', 'Drill', 'Calendar', 'Screwdriver', 'Candles', 'Curtains', 'Eraser', 'Pan',
            'Printer', 'Table', 'TV', 'Telephone', 'Push_Pin', 'Laptop', 'Chair', 'Flipflops', 'Scissors', 'Bed',
            'Paper_Clip', 'Monitor', 'Bike', 'Postit_Notes', 'Alarm_Clock', 'Webcam', 'Shelf', 'Exit_Sign', 'Notebook',
            'Oven', 'Trash_Can', 'Radio', 'Bucket', 'Folder', 'Pencil', 'Helmet'],
    'Clipart': ['Sneakers', 'Mouse', 'Flowers', 'Bottle', 'Fork', 'Fan', 'Refrigerator', 'Speaker', 'Calculator',
                'Clipboards', 'Marker', 'Batteries', 'Lamp_Shade', 'Hammer', 'Kettle', 'Mug', 'Ruler', 'Sink', 'Pen',
                'File_Cabinet', 'Backpack', 'Keyboard', 'Desk_Lamp', 'Glasses', 'Soda', 'Spoon', 'Mop', 'Computer',
                'Couch', 'ToothBrush', 'Toys', 'Knives', 'Drill', 'Calendar', 'Screwdriver', 'Candles', 'Curtains',
                'Eraser', 'Pan', 'Printer', 'Table', 'TV', 'Telephone', 'Push_Pin', 'Laptop', 'Chair', 'Flipflops',
                'Scissors', 'Bed', 'Paper_Clip', 'Monitor', 'Bike', 'Postit_Notes', 'Alarm_Clock', 'Webcam', 'Shelf',
                'Exit_Sign', 'Notebook', 'Oven', 'Trash_Can', 'Radio', 'Bucket', 'Folder', 'Pencil', 'Helmet'],
    'Real_World': ['Sneakers', 'Mouse', 'Flowers', 'Bottle', 'Fork', 'Fan', 'Refrigerator', 'Speaker', 'Calculator',
                   'Clipboards', 'Marker', 'Batteries', 'Lamp_Shade', 'Hammer', 'Kettle', 'Mug', 'Ruler', 'Sink', 'Pen',
                   'File_Cabinet', 'Backpack', 'Keyboard', 'Desk_Lamp', 'Glasses', 'Soda', 'Spoon', 'Mop', 'Computer',
                   'Couch', 'ToothBrush', 'Toys', 'Knives', 'Drill', 'Calendar', 'Screwdriver', 'Candles', 'Curtains',
                   'Eraser', 'Pan', 'Printer', 'Table', 'TV', 'Telephone', 'Push_Pin', 'Laptop', 'Chair', 'Flipflops',
                   'Scissors', 'Bed', 'Paper_Clip', 'Monitor', 'Bike', 'Postit_Notes', 'Alarm_Clock', 'Webcam', 'Shelf',
                   'Exit_Sign', 'Notebook', 'Oven', 'Trash_Can', 'Radio', 'Bucket', 'Folder', 'Pencil', 'Helmet']
}

data_settings['office-home']['APR_C']['C'] = {
    'Art': ['Sneakers', 'Mouse', 'Flowers', 'Bottle', 'Fork', 'Fan', 'Refrigerator', 'Speaker', 'Calculator',
            'Clipboards', 'Marker', 'Batteries', 'Lamp_Shade', 'Hammer', 'Kettle', 'Mug', 'Ruler', 'Sink', 'Pen',
            'File_Cabinet', 'Backpack', 'Keyboard', 'Desk_Lamp', 'Glasses', 'Soda', 'Spoon', 'Mop', 'Computer', 'Couch',
            'ToothBrush', 'Toys', 'Knives', 'Drill', 'Calendar', 'Screwdriver', 'Candles', 'Curtains', 'Eraser', 'Pan',
            'Printer', 'Table', 'TV', 'Telephone', 'Push_Pin', 'Laptop', 'Chair', 'Flipflops', 'Scissors', 'Bed',
            'Paper_Clip', 'Monitor', 'Bike', 'Postit_Notes', 'Alarm_Clock', 'Webcam', 'Shelf', 'Exit_Sign', 'Notebook',
            'Oven', 'Trash_Can', 'Radio', 'Bucket', 'Folder', 'Pencil', 'Helmet'],
    'Product': ['Sneakers', 'Mouse', 'Flowers', 'Bottle', 'Fork', 'Fan', 'Refrigerator', 'Speaker', 'Calculator',
                'Clipboards', 'Marker', 'Batteries', 'Lamp_Shade', 'Hammer', 'Kettle', 'Mug', 'Ruler', 'Sink', 'Pen',
                'File_Cabinet', 'Backpack', 'Keyboard', 'Desk_Lamp', 'Glasses', 'Soda', 'Spoon', 'Mop', 'Computer',
                'Couch', 'ToothBrush', 'Toys', 'Knives', 'Drill', 'Calendar', 'Screwdriver', 'Candles', 'Curtains',
                'Eraser', 'Pan', 'Printer', 'Table', 'TV', 'Telephone', 'Push_Pin', 'Laptop', 'Chair', 'Flipflops',
                'Scissors', 'Bed', 'Paper_Clip', 'Monitor', 'Bike', 'Postit_Notes', 'Alarm_Clock', 'Webcam', 'Shelf',
                'Exit_Sign', 'Notebook', 'Oven', 'Trash_Can', 'Radio', 'Bucket', 'Folder', 'Pencil', 'Helmet'],
    'Real_World': ['Sneakers', 'Mouse', 'Flowers', 'Bottle', 'Fork', 'Fan', 'Refrigerator', 'Speaker', 'Calculator',
                   'Clipboards', 'Marker', 'Batteries', 'Lamp_Shade', 'Hammer', 'Kettle', 'Mug', 'Ruler', 'Sink', 'Pen',
                   'File_Cabinet', 'Backpack', 'Keyboard', 'Desk_Lamp', 'Glasses', 'Soda', 'Spoon', 'Mop', 'Computer',
                   'Couch', 'ToothBrush', 'Toys', 'Knives', 'Drill', 'Calendar', 'Screwdriver', 'Candles', 'Curtains',
                   'Eraser', 'Pan', 'Printer', 'Table', 'TV', 'Telephone', 'Push_Pin', 'Laptop', 'Chair', 'Flipflops',
                   'Scissors', 'Bed', 'Paper_Clip', 'Monitor', 'Bike', 'Postit_Notes', 'Alarm_Clock', 'Webcam', 'Shelf',
                   'Exit_Sign', 'Notebook', 'Oven', 'Trash_Can', 'Radio', 'Bucket', 'Folder', 'Pencil', 'Helmet']
}

data_settings['office-home']['CPR_A']['C'] = {
    'Clipart': ['Sneakers', 'Mouse', 'Flowers', 'Bottle', 'Fork', 'Fan', 'Refrigerator', 'Speaker', 'Calculator',
                'Clipboards', 'Marker', 'Batteries', 'Lamp_Shade', 'Hammer', 'Kettle', 'Mug', 'Ruler', 'Sink', 'Pen',
                'File_Cabinet', 'Backpack', 'Keyboard', 'Desk_Lamp', 'Glasses', 'Soda', 'Spoon', 'Mop', 'Computer',
                'Couch', 'ToothBrush', 'Toys', 'Knives', 'Drill', 'Calendar', 'Screwdriver', 'Candles', 'Curtains',
                'Eraser', 'Pan', 'Printer', 'Table', 'TV', 'Telephone', 'Push_Pin', 'Laptop', 'Chair', 'Flipflops',
                'Scissors', 'Bed', 'Paper_Clip', 'Monitor', 'Bike', 'Postit_Notes', 'Alarm_Clock', 'Webcam', 'Shelf',
                'Exit_Sign', 'Notebook', 'Oven', 'Trash_Can', 'Radio', 'Bucket', 'Folder', 'Pencil', 'Helmet'],
    'Product': ['Sneakers', 'Mouse', 'Flowers', 'Bottle', 'Fork', 'Fan', 'Refrigerator', 'Speaker', 'Calculator',
                'Clipboards', 'Marker', 'Batteries', 'Lamp_Shade', 'Hammer', 'Kettle', 'Mug', 'Ruler', 'Sink', 'Pen',
                'File_Cabinet', 'Backpack', 'Keyboard', 'Desk_Lamp', 'Glasses', 'Soda', 'Spoon', 'Mop', 'Computer',
                'Couch', 'ToothBrush', 'Toys', 'Knives', 'Drill', 'Calendar', 'Screwdriver', 'Candles', 'Curtains',
                'Eraser', 'Pan', 'Printer', 'Table', 'TV', 'Telephone', 'Push_Pin', 'Laptop', 'Chair', 'Flipflops',
                'Scissors', 'Bed', 'Paper_Clip', 'Monitor', 'Bike', 'Postit_Notes', 'Alarm_Clock', 'Webcam', 'Shelf',
                'Exit_Sign', 'Notebook', 'Oven', 'Trash_Can', 'Radio', 'Bucket', 'Folder', 'Pencil', 'Helmet'],
    'Real_World': ['Sneakers', 'Mouse', 'Flowers', 'Bottle', 'Fork', 'Fan', 'Refrigerator', 'Speaker', 'Calculator',
                   'Clipboards', 'Marker', 'Batteries', 'Lamp_Shade', 'Hammer', 'Kettle', 'Mug', 'Ruler', 'Sink', 'Pen',
                   'File_Cabinet', 'Backpack', 'Keyboard', 'Desk_Lamp', 'Glasses', 'Soda', 'Spoon', 'Mop', 'Computer',
                   'Couch', 'ToothBrush', 'Toys', 'Knives', 'Drill', 'Calendar', 'Screwdriver', 'Candles', 'Curtains',
                   'Eraser', 'Pan', 'Printer', 'Table', 'TV', 'Telephone', 'Push_Pin', 'Laptop', 'Chair', 'Flipflops',
                   'Scissors', 'Bed', 'Paper_Clip', 'Monitor', 'Bike', 'Postit_Notes', 'Alarm_Clock', 'Webcam', 'Shelf',
                   'Exit_Sign', 'Notebook', 'Oven', 'Trash_Can', 'Radio', 'Bucket', 'Folder', 'Pencil', 'Helmet']
}

data_settings['office-home']['ACP_R']['C_dash'] = {
    'Clipart': [],
    'Product': [],
    'Real_World': [],
    'Art': []
}

data_settings['office-home']['ACR_P']['C_dash'] = {
    'Clipart': [],
    'Product': [],
    'Real_World': [],
    'Art': []
}

data_settings['office-home']['APR_C']['C_dash'] = {
    'Clipart': [],
    'Product': [],
    'Real_World': [],
    'Art': []
}

data_settings['office-home']['CPR_A']['C_dash'] = {
    'Clipart': [],
    'Product': [],
    'Real_World': [],
    'Art': []
}

data_settings['office-home']['ACP_R']['src_datasets'] = ['Art', 'Clipart', 'Product']
data_settings['office-home']['ACR_P']['src_datasets'] = ['Art', 'Clipart', 'Real_World']
data_settings['office-home']['APR_C']['src_datasets'] = ['Art', 'Real_World', 'Product']
data_settings['office-home']['CPR_A']['src_datasets'] = ['Real_World', 'Clipart', 'Product']

data_settings['office-home']['ACP_R']['trgt_datasets'] = ['Real_World']
data_settings['office-home']['ACR_P']['trgt_datasets'] = ['Product']
data_settings['office-home']['APR_C']['trgt_datasets'] = ['Clipart']
data_settings['office-home']['CPR_A']['trgt_datasets'] = ['Art']

data_settings['office-home']['ACP_R']['num_C'] = {'Art': len(data_settings['office-home']['ACP_R']['C']['Art']),
                                                  'Clipart': len(data_settings['office-home']['ACP_R']['C']['Clipart']),
                                                  'Product': len(data_settings['office-home']['ACP_R']['C']['Product']),
                                                  'Real_World': len(
                                                      set(data_settings['office-home']['ACP_R']['C']['Art'] +
                                                          data_settings['office-home']['ACP_R']['C']['Clipart'] +
                                                          data_settings['office-home']['ACP_R']['C']['Product']
                                                          ))
                                                  }

data_settings['office-home']['ACR_P']['num_C'] = {'Art': len(data_settings['office-home']['ACR_P']['C']['Art']),
                                                  'Clipart': len(data_settings['office-home']['ACR_P']['C']['Clipart']),
                                                  'Real_World': len(
                                                      data_settings['office-home']['ACR_P']['C']['Real_World']),
                                                  'Product': len(set(data_settings['office-home']['ACR_P']['C']['Art'] +
                                                                     data_settings['office-home']['ACR_P']['C'][
                                                                         'Clipart'] +
                                                                     data_settings['office-home']['ACR_P']['C'][
                                                                         'Real_World']
                                                                     ))
                                                  }

data_settings['office-home']['APR_C']['num_C'] = {'Art': len(data_settings['office-home']['APR_C']['C']['Art']),
                                                  'Real_World': len(
                                                      data_settings['office-home']['APR_C']['C']['Real_World']),
                                                  'Product': len(data_settings['office-home']['APR_C']['C']['Product']),
                                                  'Clipart': len(set(data_settings['office-home']['APR_C']['C']['Art'] +
                                                                     data_settings['office-home']['APR_C']['C'][
                                                                         'Product'] +
                                                                     data_settings['office-home']['APR_C']['C'][
                                                                         'Real_World']
                                                                     ))
                                                  }

data_settings['office-home']['CPR_A']['num_C'] = {'Clipart': len(data_settings['office-home']['CPR_A']['C']['Clipart']),
                                                  'Real_World': len(
                                                      data_settings['office-home']['CPR_A']['C']['Real_World']),
                                                  'Product': len(data_settings['office-home']['CPR_A']['C']['Product']),
                                                  'Art': len(set(data_settings['office-home']['CPR_A']['C']['Clipart'] +
                                                                 data_settings['office-home']['CPR_A']['C']['Product'] +
                                                                 data_settings['office-home']['CPR_A']['C'][
                                                                     'Real_World']
                                                                 ))
                                                  }

data_settings['office-home']['ACP_R']['num_C_dash'] = {
    'Art': len(data_settings['office-home']['ACP_R']['C_dash']['Art']),
    'Clipart': len(data_settings['office-home']['ACP_R']['C_dash']['Clipart']),
    'Product': len(data_settings['office-home']['ACP_R']['C_dash']['Product']),
    'Real_World': len(data_settings['office-home']['ACP_R']['C_dash']['Real_World'])
    }

data_settings['office-home']['ACR_P']['num_C_dash'] = {
    'Art': len(data_settings['office-home']['ACR_P']['C_dash']['Art']),
    'Clipart': len(data_settings['office-home']['ACR_P']['C_dash']['Clipart']),
    'Product': len(data_settings['office-home']['ACR_P']['C_dash']['Product']),
    'Real_World': len(data_settings['office-home']['ACR_P']['C_dash']['Real_World'])
    }

data_settings['office-home']['APR_C']['num_C_dash'] = {
    'Art': len(data_settings['office-home']['APR_C']['C_dash']['Art']),
    'Clipart': len(data_settings['office-home']['APR_C']['C_dash']['Clipart']),
    'Product': len(data_settings['office-home']['APR_C']['C_dash']['Product']),
    'Real_World': len(data_settings['office-home']['APR_C']['C_dash']['Real_World'])
    }

data_settings['office-home']['CPR_A']['num_C_dash'] = {
    'Art': len(data_settings['office-home']['CPR_A']['C_dash']['Art']),
    'Clipart': len(data_settings['office-home']['CPR_A']['C_dash']['Clipart']),
    'Product': len(data_settings['office-home']['CPR_A']['C_dash']['Product']),
    'Real_World': len(data_settings['office-home']['CPR_A']['C_dash']['Real_World'])
    }

data_settings['domain-net']['CIPQS_R']['C'] = {
    'clipart': ['raccoon', 'windmill', 'lightning', 'mouth', 'animal_migration', 'hammer', 'bathtub', 'television',
                'birthday_cake', 'arm', 'telephone', 'firetruck', 'toothbrush', 'sailboat', 'river', 'baseball',
                'scorpion', 'golf_club', 'ear', 'traffic_light', 'frying_pan', 'door', 'frog', 'keyboard', 'zebra',
                'bread', 'umbrella', 'onion', 'stereo', 'dishwasher', 'pants', 'crown', 'hospital', 'square', 't-shirt',
                'nail', 'squirrel', 'couch', 'hockey_puck', 'laptop', 'dog', 'steak', 'violin', 'cooler', 'bracelet',
                'crab', 'cell_phone', 'shoe', 'ant', 'asparagus', 'garden_hose', 'rain', 'pineapple', 'table', 'rabbit',
                'pickup_truck', 'peas', 'ice_cream', 'The_Eiffel_Tower', 'clarinet', 'string_bean', 'hedgehog',
                'rollerskates', 'ceiling_fan', 'butterfly', 'flashlight', 'crayon', 'palm_tree', 'truck', 'shark',
                'cat', 'binoculars', 'bee', 'star', 'foot', 'smiley_face', 'envelope', 'teddy-bear', 'feather', 'tooth',
                'lion', 'toe', 'hand', 'boomerang', 'tree', 'sword', 'diamond', 'pool', 'hamburger', 'chandelier',
                'floor_lamp', 'circle', 'kangaroo', 'hat', 'snake', 'house', 'sea_turtle', 'campfire', 'soccer_ball',
                'megaphone', 'grass', 'jacket', 'mountain', 'cookie', 'wine_glass', 'octagon', 'church', 'cruise_ship',
                'stop_sign', 'knife', 'belt', 'hurricane', 'piano', 'pear', 'wheel', 'castle', 'sink', 'headphones',
                'bus', 'tennis_racquet', 'shovel', 'moon', 'hot_tub', 'trombone', 'pencil', 'duck', 'knee',
                'blackberry', 'panda', 'garden', 'chair', 'tractor', 'house_plant', 'spreadsheet', 'skateboard',
                'bandage', 'tornado', 'anvil', 'squiggle', 'mosquito', 'jail', 'snowflake', 'sock', 'bed', 'vase',
                'baseball_bat', 'popsicle', 'bowtie', 'calculator', 'microphone', 'computer', 'wristwatch', 'shorts',
                'dresser', 'mermaid', 'hexagon', 'eyeglasses', 'bulldozer', 'harp', 'fish', 'line', 'pliers',
                'speedboat', 'toaster', 'banana', 'purse', 'broccoli', 'toothpaste', 'dolphin', 'bottlecap',
                'washing_machine', 'bird', 'pig', 'pillow', 'parrot', 'giraffe', 'lighthouse', 'cello', 'swan', 'donut',
                'lantern', 'postcard', 'eye', 'finger', 'coffee_cup', 'aircraft_carrier', 'horse', 'paint_can',
                'lollipop', 'snowman', 'skyscraper', 'stitches', 'lipstick', 'camouflage', 'book', 'rake',
                'hot_air_balloon', 'saxophone', 'map', 'matches', 'tiger', 'bat', 'cannon', 'hockey_stick',
                'power_outlet', 'screwdriver', 'marker', 'see_saw', 'barn', 'bucket', 'ladder', 'mouse', 'underwear',
                'monkey', 'leg', 'train', 'mailbox', 'basketball', 'moustache', 'microwave', 'cactus', 'rainbow',
                'fireplace', 'sheep', 'tent', 'fan', 'police_car', 'sandwich', 'sleeping_bag', 'guitar', 'mug', 'face',
                'compass', 'spoon', 'broom', 'alarm_clock', 'snorkel', 'nose', 'hourglass', 'drill', 'cloud', 'cake',
                'skull', 'saw', 'goatee', 'cup', 'dumbbell', 'apple', 'pizza', 'canoe', 'bicycle', 'grapes', 'toilet',
                'blueberry', 'submarine', 'backpack', 'stethoscope', 'airplane', 'drums', 'fence', 'teapot',
                'remote_control', 'calendar', 'suitcase', 'wine_bottle', 'helicopter', 'hot_dog', 'car', 'owl',
                'mushroom', 'school_bus', 'whale', 'crocodile', 'roller_coaster', 'octopus', 'basket', 'potato', 'leaf',
                'zigzag', 'syringe', 'pond', 'dragon', 'triangle', 'carrot', 'parachute', 'van', 'stove', 'bridge',
                'swing_set', 'bush', 'The_Great_Wall_of_China', 'helmet', 'flying_saucer', 'trumpet', 'penguin',
                'candle', 'peanut', 'beard', 'fire_hydrant', 'cow', 'lighter', 'waterslide', 'elephant', 'oven',
                'flower', 'spider', 'motorbike', 'strawberry', 'diving_board', 'sweater', 'paper_clip', 'scissors',
                'angel', 'key', 'yoga', 'fork', 'axe', 'rhinoceros', 'brain', 'flip_flops', 'elbow', 'light_bulb',
                'radio', 'camel', 'paintbrush', 'watermelon', 'The_Mona_Lisa', 'stairs', 'picture_frame', 'clock',
                'sun', 'bear', 'bench', 'camera', 'eraser', 'snail', 'beach', 'ocean', 'rifle', 'passport', 'ambulance',
                'lobster', 'flamingo', 'streetlight', 'necklace'],
    'infograph': ['raccoon', 'windmill', 'lightning', 'mouth', 'animal_migration', 'hammer', 'bathtub', 'television',
                  'birthday_cake', 'arm', 'telephone', 'firetruck', 'toothbrush', 'sailboat', 'river', 'baseball',
                  'scorpion', 'golf_club', 'ear', 'traffic_light', 'frying_pan', 'door', 'frog', 'keyboard', 'zebra',
                  'bread', 'umbrella', 'onion', 'stereo', 'dishwasher', 'pants', 'crown', 'hospital', 'square',
                  't-shirt', 'nail', 'squirrel', 'couch', 'hockey_puck', 'laptop', 'dog', 'steak', 'violin', 'cooler',
                  'bracelet', 'crab', 'cell_phone', 'shoe', 'ant', 'asparagus', 'garden_hose', 'rain', 'pineapple',
                  'table', 'rabbit', 'pickup_truck', 'peas', 'ice_cream', 'The_Eiffel_Tower', 'clarinet', 'string_bean',
                  'hedgehog', 'rollerskates', 'ceiling_fan', 'butterfly', 'flashlight', 'crayon', 'palm_tree', 'truck',
                  'shark', 'cat', 'binoculars', 'bee', 'star', 'foot', 'smiley_face', 'envelope', 'teddy-bear',
                  'feather', 'tooth', 'lion', 'toe', 'hand', 'boomerang', 'tree', 'sword', 'diamond', 'pool',
                  'hamburger', 'chandelier', 'floor_lamp', 'circle', 'kangaroo', 'hat', 'snake', 'house', 'sea_turtle',
                  'campfire', 'soccer_ball', 'megaphone', 'grass', 'jacket', 'mountain', 'cookie', 'wine_glass',
                  'octagon', 'church', 'cruise_ship', 'stop_sign', 'knife', 'belt', 'hurricane', 'piano', 'pear',
                  'wheel', 'castle', 'sink', 'headphones', 'bus', 'tennis_racquet', 'shovel', 'moon', 'hot_tub',
                  'trombone', 'pencil', 'duck', 'knee', 'blackberry', 'panda', 'garden', 'chair', 'tractor',
                  'house_plant', 'spreadsheet', 'skateboard', 'bandage', 'tornado', 'anvil', 'squiggle', 'mosquito',
                  'jail', 'snowflake', 'sock', 'bed', 'vase', 'baseball_bat', 'popsicle', 'bowtie', 'calculator',
                  'microphone', 'computer', 'wristwatch', 'shorts', 'dresser', 'mermaid', 'hexagon', 'eyeglasses',
                  'bulldozer', 'harp', 'fish', 'line', 'pliers', 'speedboat', 'toaster', 'banana', 'purse', 'broccoli',
                  'toothpaste', 'dolphin', 'bottlecap', 'washing_machine', 'bird', 'pig', 'pillow', 'parrot', 'giraffe',
                  'lighthouse', 'cello', 'swan', 'donut', 'lantern', 'postcard', 'eye', 'finger', 'coffee_cup',
                  'aircraft_carrier', 'horse', 'paint_can', 'lollipop', 'snowman', 'skyscraper', 'stitches', 'lipstick',
                  'camouflage', 'book', 'rake', 'hot_air_balloon', 'saxophone', 'map', 'matches', 'tiger', 'bat',
                  'cannon', 'hockey_stick', 'power_outlet', 'screwdriver', 'marker', 'see_saw', 'barn', 'bucket',
                  'ladder', 'mouse', 'underwear', 'monkey', 'leg', 'train', 'mailbox', 'basketball', 'moustache',
                  'microwave', 'cactus', 'rainbow', 'fireplace', 'sheep', 'tent', 'fan', 'police_car', 'sandwich',
                  'sleeping_bag', 'guitar', 'mug', 'face', 'compass', 'spoon', 'broom', 'alarm_clock', 'snorkel',
                  'nose', 'hourglass', 'drill', 'cloud', 'cake', 'skull', 'saw', 'goatee', 'cup', 'dumbbell', 'apple',
                  'pizza', 'canoe', 'bicycle', 'grapes', 'toilet', 'blueberry', 'submarine', 'backpack', 'stethoscope',
                  'airplane', 'drums', 'fence', 'teapot', 'remote_control', 'calendar', 'suitcase', 'wine_bottle',
                  'helicopter', 'hot_dog', 'car', 'owl', 'mushroom', 'school_bus', 'whale', 'crocodile',
                  'roller_coaster', 'octopus', 'basket', 'potato', 'leaf', 'zigzag', 'syringe', 'pond', 'dragon',
                  'triangle', 'carrot', 'parachute', 'van', 'stove', 'bridge', 'swing_set', 'bush',
                  'The_Great_Wall_of_China', 'helmet', 'flying_saucer', 'trumpet', 'penguin', 'candle', 'peanut',
                  'beard', 'fire_hydrant', 'cow', 'lighter', 'waterslide', 'elephant', 'oven', 'flower', 'spider',
                  'motorbike', 'strawberry', 'diving_board', 'sweater', 'paper_clip', 'scissors', 'angel', 'key',
                  'yoga', 'fork', 'axe', 'rhinoceros', 'brain', 'flip_flops', 'elbow', 'light_bulb', 'radio', 'camel',
                  'paintbrush', 'watermelon', 'The_Mona_Lisa', 'stairs', 'picture_frame', 'clock', 'sun', 'bear',
                  'bench', 'camera', 'eraser', 'snail', 'beach', 'ocean', 'rifle', 'passport', 'ambulance', 'lobster',
                  'flamingo', 'streetlight', 'necklace'],
    'painting': ['raccoon', 'windmill', 'lightning', 'mouth', 'animal_migration', 'hammer', 'bathtub', 'television',
                 'birthday_cake', 'arm', 'telephone', 'firetruck', 'toothbrush', 'sailboat', 'river', 'baseball',
                 'scorpion', 'golf_club', 'ear', 'traffic_light', 'frying_pan', 'door', 'frog', 'keyboard', 'zebra',
                 'bread', 'umbrella', 'onion', 'stereo', 'dishwasher', 'pants', 'crown', 'hospital', 'square',
                 't-shirt', 'nail', 'squirrel', 'couch', 'hockey_puck', 'laptop', 'dog', 'steak', 'violin', 'cooler',
                 'bracelet', 'crab', 'cell_phone', 'shoe', 'ant', 'asparagus', 'garden_hose', 'rain', 'pineapple',
                 'table', 'rabbit', 'pickup_truck', 'peas', 'ice_cream', 'The_Eiffel_Tower', 'clarinet', 'string_bean',
                 'hedgehog', 'rollerskates', 'ceiling_fan', 'butterfly', 'flashlight', 'crayon', 'palm_tree', 'truck',
                 'shark', 'cat', 'binoculars', 'bee', 'star', 'foot', 'smiley_face', 'envelope', 'teddy-bear',
                 'feather', 'tooth', 'lion', 'toe', 'hand', 'boomerang', 'tree', 'sword', 'diamond', 'pool',
                 'hamburger', 'chandelier', 'floor_lamp', 'circle', 'kangaroo', 'hat', 'snake', 'house', 'sea_turtle',
                 'campfire', 'soccer_ball', 'megaphone', 'grass', 'jacket', 'mountain', 'cookie', 'wine_glass',
                 'octagon', 'church', 'cruise_ship', 'stop_sign', 'knife', 'belt', 'hurricane', 'piano', 'pear',
                 'wheel', 'castle', 'sink', 'headphones', 'bus', 'tennis_racquet', 'shovel', 'moon', 'hot_tub',
                 'trombone', 'pencil', 'duck', 'knee', 'blackberry', 'panda', 'garden', 'chair', 'tractor',
                 'house_plant', 'spreadsheet', 'skateboard', 'bandage', 'tornado', 'anvil', 'squiggle', 'mosquito',
                 'jail', 'snowflake', 'sock', 'bed', 'vase', 'baseball_bat', 'popsicle', 'bowtie', 'calculator',
                 'microphone', 'computer', 'wristwatch', 'shorts', 'dresser', 'mermaid', 'hexagon', 'eyeglasses',
                 'bulldozer', 'harp', 'fish', 'line', 'pliers', 'speedboat', 'toaster', 'banana', 'purse', 'broccoli',
                 'toothpaste', 'dolphin', 'bottlecap', 'washing_machine', 'bird', 'pig', 'pillow', 'parrot', 'giraffe',
                 'lighthouse', 'cello', 'swan', 'donut', 'lantern', 'postcard', 'eye', 'finger', 'coffee_cup',
                 'aircraft_carrier', 'horse', 'paint_can', 'lollipop', 'snowman', 'skyscraper', 'stitches', 'lipstick',
                 'camouflage', 'book', 'rake', 'hot_air_balloon', 'saxophone', 'map', 'matches', 'tiger', 'bat',
                 'cannon', 'hockey_stick', 'power_outlet', 'screwdriver', 'marker', 'see_saw', 'barn', 'bucket',
                 'ladder', 'mouse', 'underwear', 'monkey', 'leg', 'train', 'mailbox', 'basketball', 'moustache',
                 'microwave', 'cactus', 'rainbow', 'fireplace', 'sheep', 'tent', 'fan', 'police_car', 'sandwich',
                 'sleeping_bag', 'guitar', 'mug', 'face', 'compass', 'spoon', 'broom', 'alarm_clock', 'snorkel', 'nose',
                 'hourglass', 'drill', 'cloud', 'cake', 'skull', 'saw', 'goatee', 'cup', 'dumbbell', 'apple', 'pizza',
                 'canoe', 'bicycle', 'grapes', 'toilet', 'blueberry', 'submarine', 'backpack', 'stethoscope',
                 'airplane', 'drums', 'fence', 'teapot', 'remote_control', 'calendar', 'suitcase', 'wine_bottle',
                 'helicopter', 'hot_dog', 'car', 'owl', 'mushroom', 'school_bus', 'whale', 'crocodile',
                 'roller_coaster', 'octopus', 'basket', 'potato', 'leaf', 'zigzag', 'syringe', 'pond', 'dragon',
                 'triangle', 'carrot', 'parachute', 'van', 'stove', 'bridge', 'swing_set', 'bush',
                 'The_Great_Wall_of_China', 'helmet', 'flying_saucer', 'trumpet', 'penguin', 'candle', 'peanut',
                 'beard', 'fire_hydrant', 'cow', 'lighter', 'waterslide', 'elephant', 'oven', 'flower', 'spider',
                 'motorbike', 'strawberry', 'diving_board', 'sweater', 'paper_clip', 'scissors', 'angel', 'key', 'yoga',
                 'fork', 'axe', 'rhinoceros', 'brain', 'flip_flops', 'elbow', 'light_bulb', 'radio', 'camel',
                 'paintbrush', 'watermelon', 'The_Mona_Lisa', 'stairs', 'picture_frame', 'clock', 'sun', 'bear',
                 'bench', 'camera', 'eraser', 'snail', 'beach', 'ocean', 'rifle', 'passport', 'ambulance', 'lobster',
                 'flamingo', 'streetlight', 'necklace'],
    'quickdraw': ['raccoon', 'windmill', 'lightning', 'mouth', 'animal_migration', 'hammer', 'bathtub', 'television',
                  'birthday_cake', 'arm', 'telephone', 'firetruck', 'toothbrush', 'sailboat', 'river', 'baseball',
                  'scorpion', 'golf_club', 'ear', 'traffic_light', 'frying_pan', 'door', 'frog', 'keyboard', 'zebra',
                  'bread', 'umbrella', 'onion', 'stereo', 'dishwasher', 'pants', 'crown', 'hospital', 'square',
                  't-shirt', 'nail', 'squirrel', 'couch', 'hockey_puck', 'laptop', 'dog', 'steak', 'violin', 'cooler',
                  'bracelet', 'crab', 'cell_phone', 'shoe', 'ant', 'asparagus', 'garden_hose', 'rain', 'pineapple',
                  'table', 'rabbit', 'pickup_truck', 'peas', 'ice_cream', 'The_Eiffel_Tower', 'clarinet', 'string_bean',
                  'hedgehog', 'rollerskates', 'ceiling_fan', 'butterfly', 'flashlight', 'crayon', 'palm_tree', 'truck',
                  'shark', 'cat', 'binoculars', 'bee', 'star', 'foot', 'smiley_face', 'envelope', 'teddy-bear',
                  'feather', 'tooth', 'lion', 'toe', 'hand', 'boomerang', 'tree', 'sword', 'diamond', 'pool',
                  'hamburger', 'chandelier', 'floor_lamp', 'circle', 'kangaroo', 'hat', 'snake', 'house', 'sea_turtle',
                  'campfire', 'soccer_ball', 'megaphone', 'grass', 'jacket', 'mountain', 'cookie', 'wine_glass',
                  'octagon', 'church', 'cruise_ship', 'stop_sign', 'knife', 'belt', 'hurricane', 'piano', 'pear',
                  'wheel', 'castle', 'sink', 'headphones', 'bus', 'tennis_racquet', 'shovel', 'moon', 'hot_tub',
                  'trombone', 'pencil', 'duck', 'knee', 'blackberry', 'panda', 'garden', 'chair', 'tractor',
                  'house_plant', 'spreadsheet', 'skateboard', 'bandage', 'tornado', 'anvil', 'squiggle', 'mosquito',
                  'jail', 'snowflake', 'sock', 'bed', 'vase', 'baseball_bat', 'popsicle', 'bowtie', 'calculator',
                  'microphone', 'computer', 'wristwatch', 'shorts', 'dresser', 'mermaid', 'hexagon', 'eyeglasses',
                  'bulldozer', 'harp', 'fish', 'line', 'pliers', 'speedboat', 'toaster', 'banana', 'purse', 'broccoli',
                  'toothpaste', 'dolphin', 'bottlecap', 'washing_machine', 'bird', 'pig', 'pillow', 'parrot', 'giraffe',
                  'lighthouse', 'cello', 'swan', 'donut', 'lantern', 'postcard', 'eye', 'finger', 'coffee_cup',
                  'aircraft_carrier', 'horse', 'paint_can', 'lollipop', 'snowman', 'skyscraper', 'stitches', 'lipstick',
                  'camouflage', 'book', 'rake', 'hot_air_balloon', 'saxophone', 'map', 'matches', 'tiger', 'bat',
                  'cannon', 'hockey_stick', 'power_outlet', 'screwdriver', 'marker', 'see_saw', 'barn', 'bucket',
                  'ladder', 'mouse', 'underwear', 'monkey', 'leg', 'train', 'mailbox', 'basketball', 'moustache',
                  'microwave', 'cactus', 'rainbow', 'fireplace', 'sheep', 'tent', 'fan', 'police_car', 'sandwich',
                  'sleeping_bag', 'guitar', 'mug', 'face', 'compass', 'spoon', 'broom', 'alarm_clock', 'snorkel',
                  'nose', 'hourglass', 'drill', 'cloud', 'cake', 'skull', 'saw', 'goatee', 'cup', 'dumbbell', 'apple',
                  'pizza', 'canoe', 'bicycle', 'grapes', 'toilet', 'blueberry', 'submarine', 'backpack', 'stethoscope',
                  'airplane', 'drums', 'fence', 'teapot', 'remote_control', 'calendar', 'suitcase', 'wine_bottle',
                  'helicopter', 'hot_dog', 'car', 'owl', 'mushroom', 'school_bus', 'whale', 'crocodile',
                  'roller_coaster', 'octopus', 'basket', 'potato', 'leaf', 'zigzag', 'syringe', 'pond', 'dragon',
                  'triangle', 'carrot', 'parachute', 'van', 'stove', 'bridge', 'swing_set', 'bush',
                  'The_Great_Wall_of_China', 'helmet', 'flying_saucer', 'trumpet', 'penguin', 'candle', 'peanut',
                  'beard', 'fire_hydrant', 'cow', 'lighter', 'waterslide', 'elephant', 'oven', 'flower', 'spider',
                  'motorbike', 'strawberry', 'diving_board', 'sweater', 'paper_clip', 'scissors', 'angel', 'key',
                  'yoga', 'fork', 'axe', 'rhinoceros', 'brain', 'flip_flops', 'elbow', 'light_bulb', 'radio', 'camel',
                  'paintbrush', 'watermelon', 'The_Mona_Lisa', 'stairs', 'picture_frame', 'clock', 'sun', 'bear',
                  'bench', 'camera', 'eraser', 'snail', 'beach', 'ocean', 'rifle', 'passport', 'ambulance', 'lobster',
                  'flamingo', 'streetlight', 'necklace'],
    'sketch': ['raccoon', 'windmill', 'lightning', 'mouth', 'animal_migration', 'hammer', 'bathtub', 'television',
               'birthday_cake', 'arm', 'telephone', 'firetruck', 'toothbrush', 'sailboat', 'river', 'baseball',
               'scorpion', 'golf_club', 'ear', 'traffic_light', 'frying_pan', 'door', 'frog', 'keyboard', 'zebra',
               'bread', 'umbrella', 'onion', 'stereo', 'dishwasher', 'pants', 'crown', 'hospital', 'square', 't-shirt',
               'nail', 'squirrel', 'couch', 'hockey_puck', 'laptop', 'dog', 'steak', 'violin', 'cooler', 'bracelet',
               'crab', 'cell_phone', 'shoe', 'ant', 'asparagus', 'garden_hose', 'rain', 'pineapple', 'table', 'rabbit',
               'pickup_truck', 'peas', 'ice_cream', 'The_Eiffel_Tower', 'clarinet', 'string_bean', 'hedgehog',
               'rollerskates', 'ceiling_fan', 'butterfly', 'flashlight', 'crayon', 'palm_tree', 'truck', 'shark', 'cat',
               'binoculars', 'bee', 'star', 'foot', 'smiley_face', 'envelope', 'teddy-bear', 'feather', 'tooth', 'lion',
               'toe', 'hand', 'boomerang', 'tree', 'sword', 'diamond', 'pool', 'hamburger', 'chandelier', 'floor_lamp',
               'circle', 'kangaroo', 'hat', 'snake', 'house', 'sea_turtle', 'campfire', 'soccer_ball', 'megaphone',
               'grass', 'jacket', 'mountain', 'cookie', 'wine_glass', 'octagon', 'church', 'cruise_ship', 'stop_sign',
               'knife', 'belt', 'hurricane', 'piano', 'pear', 'wheel', 'castle', 'sink', 'headphones', 'bus',
               'tennis_racquet', 'shovel', 'moon', 'hot_tub', 'trombone', 'pencil', 'duck', 'knee', 'blackberry',
               'panda', 'garden', 'chair', 'tractor', 'house_plant', 'spreadsheet', 'skateboard', 'bandage', 'tornado',
               'anvil', 'squiggle', 'mosquito', 'jail', 'snowflake', 'sock', 'bed', 'vase', 'baseball_bat', 'popsicle',
               'bowtie', 'calculator', 'microphone', 'computer', 'wristwatch', 'shorts', 'dresser', 'mermaid',
               'hexagon', 'eyeglasses', 'bulldozer', 'harp', 'fish', 'line', 'pliers', 'speedboat', 'toaster', 'banana',
               'purse', 'broccoli', 'toothpaste', 'dolphin', 'bottlecap', 'washing_machine', 'bird', 'pig', 'pillow',
               'parrot', 'giraffe', 'lighthouse', 'cello', 'swan', 'donut', 'lantern', 'postcard', 'eye', 'finger',
               'coffee_cup', 'aircraft_carrier', 'horse', 'paint_can', 'lollipop', 'snowman', 'skyscraper', 'stitches',
               'lipstick', 'camouflage', 'book', 'rake', 'hot_air_balloon', 'saxophone', 'map', 'matches', 'tiger',
               'bat', 'cannon', 'hockey_stick', 'power_outlet', 'screwdriver', 'marker', 'see_saw', 'barn', 'bucket',
               'ladder', 'mouse', 'underwear', 'monkey', 'leg', 'train', 'mailbox', 'basketball', 'moustache',
               'microwave', 'cactus', 'rainbow', 'fireplace', 'sheep', 'tent', 'fan', 'police_car', 'sandwich',
               'sleeping_bag', 'guitar', 'mug', 'face', 'compass', 'spoon', 'broom', 'alarm_clock', 'snorkel', 'nose',
               'hourglass', 'drill', 'cloud', 'cake', 'skull', 'saw', 'goatee', 'cup', 'dumbbell', 'apple', 'pizza',
               'canoe', 'bicycle', 'grapes', 'toilet', 'blueberry', 'submarine', 'backpack', 'stethoscope', 'airplane',
               'drums', 'fence', 'teapot', 'remote_control', 'calendar', 'suitcase', 'wine_bottle', 'helicopter',
               'hot_dog', 'car', 'owl', 'mushroom', 'school_bus', 'whale', 'crocodile', 'roller_coaster', 'octopus',
               'basket', 'potato', 'leaf', 'zigzag', 'syringe', 'pond', 'dragon', 'triangle', 'carrot', 'parachute',
               'van', 'stove', 'bridge', 'swing_set', 'bush', 'The_Great_Wall_of_China', 'helmet', 'flying_saucer',
               'trumpet', 'penguin', 'candle', 'peanut', 'beard', 'fire_hydrant', 'cow', 'lighter', 'waterslide',
               'elephant', 'oven', 'flower', 'spider', 'motorbike', 'strawberry', 'diving_board', 'sweater',
               'paper_clip', 'scissors', 'angel', 'key', 'yoga', 'fork', 'axe', 'rhinoceros', 'brain', 'flip_flops',
               'elbow', 'light_bulb', 'radio', 'camel', 'paintbrush', 'watermelon', 'The_Mona_Lisa', 'stairs',
               'picture_frame', 'clock', 'sun', 'bear', 'bench', 'camera', 'eraser', 'snail', 'beach', 'ocean', 'rifle',
               'passport', 'ambulance', 'lobster', 'flamingo', 'streetlight', 'necklace']
    }

data_settings['domain-net']['CIPSR_Q']['C'] = {
    'clipart': ['raccoon', 'windmill', 'lightning', 'mouth', 'animal_migration', 'hammer', 'bathtub', 'television',
                'birthday_cake', 'arm', 'telephone', 'firetruck', 'toothbrush', 'sailboat', 'river', 'baseball',
                'scorpion', 'golf_club', 'ear', 'traffic_light', 'frying_pan', 'door', 'frog', 'keyboard', 'zebra',
                'bread', 'umbrella', 'onion', 'stereo', 'dishwasher', 'pants', 'crown', 'hospital', 'square', 't-shirt',
                'nail', 'squirrel', 'couch', 'hockey_puck', 'laptop', 'dog', 'steak', 'violin', 'cooler', 'bracelet',
                'crab', 'cell_phone', 'shoe', 'ant', 'asparagus', 'garden_hose', 'rain', 'pineapple', 'table', 'rabbit',
                'pickup_truck', 'peas', 'ice_cream', 'The_Eiffel_Tower', 'clarinet', 'string_bean', 'hedgehog',
                'rollerskates', 'ceiling_fan', 'butterfly', 'flashlight', 'crayon', 'palm_tree', 'truck', 'shark',
                'cat', 'binoculars', 'bee', 'star', 'foot', 'smiley_face', 'envelope', 'teddy-bear', 'feather', 'tooth',
                'lion', 'toe', 'hand', 'boomerang', 'tree', 'sword', 'diamond', 'pool', 'hamburger', 'chandelier',
                'floor_lamp', 'circle', 'kangaroo', 'hat', 'snake', 'house', 'sea_turtle', 'campfire', 'soccer_ball',
                'megaphone', 'grass', 'jacket', 'mountain', 'cookie', 'wine_glass', 'octagon', 'church', 'cruise_ship',
                'stop_sign', 'knife', 'belt', 'hurricane', 'piano', 'pear', 'wheel', 'castle', 'sink', 'headphones',
                'bus', 'tennis_racquet', 'shovel', 'moon', 'hot_tub', 'trombone', 'pencil', 'duck', 'knee',
                'blackberry', 'panda', 'garden', 'chair', 'tractor', 'house_plant', 'spreadsheet', 'skateboard',
                'bandage', 'tornado', 'anvil', 'squiggle', 'mosquito', 'jail', 'snowflake', 'sock', 'bed', 'vase',
                'baseball_bat', 'popsicle', 'bowtie', 'calculator', 'microphone', 'computer', 'wristwatch', 'shorts',
                'dresser', 'mermaid', 'hexagon', 'eyeglasses', 'bulldozer', 'harp', 'fish', 'line', 'pliers',
                'speedboat', 'toaster', 'banana', 'purse', 'broccoli', 'toothpaste', 'dolphin', 'bottlecap',
                'washing_machine', 'bird', 'pig', 'pillow', 'parrot', 'giraffe', 'lighthouse', 'cello', 'swan', 'donut',
                'lantern', 'postcard', 'eye', 'finger', 'coffee_cup', 'aircraft_carrier', 'horse', 'paint_can',
                'lollipop', 'snowman', 'skyscraper', 'stitches', 'lipstick', 'camouflage', 'book', 'rake',
                'hot_air_balloon', 'saxophone', 'map', 'matches', 'tiger', 'bat', 'cannon', 'hockey_stick',
                'power_outlet', 'screwdriver', 'marker', 'see_saw', 'barn', 'bucket', 'ladder', 'mouse', 'underwear',
                'monkey', 'leg', 'train', 'mailbox', 'basketball', 'moustache', 'microwave', 'cactus', 'rainbow',
                'fireplace', 'sheep', 'tent', 'fan', 'police_car', 'sandwich', 'sleeping_bag', 'guitar', 'mug', 'face',
                'compass', 'spoon', 'broom', 'alarm_clock', 'snorkel', 'nose', 'hourglass', 'drill', 'cloud', 'cake',
                'skull', 'saw', 'goatee', 'cup', 'dumbbell', 'apple', 'pizza', 'canoe', 'bicycle', 'grapes', 'toilet',
                'blueberry', 'submarine', 'backpack', 'stethoscope', 'airplane', 'drums', 'fence', 'teapot',
                'remote_control', 'calendar', 'suitcase', 'wine_bottle', 'helicopter', 'hot_dog', 'car', 'owl',
                'mushroom', 'school_bus', 'whale', 'crocodile', 'roller_coaster', 'octopus', 'basket', 'potato', 'leaf',
                'zigzag', 'syringe', 'pond', 'dragon', 'triangle', 'carrot', 'parachute', 'van', 'stove', 'bridge',
                'swing_set', 'bush', 'The_Great_Wall_of_China', 'helmet', 'flying_saucer', 'trumpet', 'penguin',
                'candle', 'peanut', 'beard', 'fire_hydrant', 'cow', 'lighter', 'waterslide', 'elephant', 'oven',
                'flower', 'spider', 'motorbike', 'strawberry', 'diving_board', 'sweater', 'paper_clip', 'scissors',
                'angel', 'key', 'yoga', 'fork', 'axe', 'rhinoceros', 'brain', 'flip_flops', 'elbow', 'light_bulb',
                'radio', 'camel', 'paintbrush', 'watermelon', 'The_Mona_Lisa', 'stairs', 'picture_frame', 'clock',
                'sun', 'bear', 'bench', 'camera', 'eraser', 'snail', 'beach', 'ocean', 'rifle', 'passport', 'ambulance',
                'lobster', 'flamingo', 'streetlight', 'necklace'],
    'infograph': ['raccoon', 'windmill', 'lightning', 'mouth', 'animal_migration', 'hammer', 'bathtub', 'television',
                  'birthday_cake', 'arm', 'telephone', 'firetruck', 'toothbrush', 'sailboat', 'river', 'baseball',
                  'scorpion', 'golf_club', 'ear', 'traffic_light', 'frying_pan', 'door', 'frog', 'keyboard', 'zebra',
                  'bread', 'umbrella', 'onion', 'stereo', 'dishwasher', 'pants', 'crown', 'hospital', 'square',
                  't-shirt', 'nail', 'squirrel', 'couch', 'hockey_puck', 'laptop', 'dog', 'steak', 'violin', 'cooler',
                  'bracelet', 'crab', 'cell_phone', 'shoe', 'ant', 'asparagus', 'garden_hose', 'rain', 'pineapple',
                  'table', 'rabbit', 'pickup_truck', 'peas', 'ice_cream', 'The_Eiffel_Tower', 'clarinet', 'string_bean',
                  'hedgehog', 'rollerskates', 'ceiling_fan', 'butterfly', 'flashlight', 'crayon', 'palm_tree', 'truck',
                  'shark', 'cat', 'binoculars', 'bee', 'star', 'foot', 'smiley_face', 'envelope', 'teddy-bear',
                  'feather', 'tooth', 'lion', 'toe', 'hand', 'boomerang', 'tree', 'sword', 'diamond', 'pool',
                  'hamburger', 'chandelier', 'floor_lamp', 'circle', 'kangaroo', 'hat', 'snake', 'house', 'sea_turtle',
                  'campfire', 'soccer_ball', 'megaphone', 'grass', 'jacket', 'mountain', 'cookie', 'wine_glass',
                  'octagon', 'church', 'cruise_ship', 'stop_sign', 'knife', 'belt', 'hurricane', 'piano', 'pear',
                  'wheel', 'castle', 'sink', 'headphones', 'bus', 'tennis_racquet', 'shovel', 'moon', 'hot_tub',
                  'trombone', 'pencil', 'duck', 'knee', 'blackberry', 'panda', 'garden', 'chair', 'tractor',
                  'house_plant', 'spreadsheet', 'skateboard', 'bandage', 'tornado', 'anvil', 'squiggle', 'mosquito',
                  'jail', 'snowflake', 'sock', 'bed', 'vase', 'baseball_bat', 'popsicle', 'bowtie', 'calculator',
                  'microphone', 'computer', 'wristwatch', 'shorts', 'dresser', 'mermaid', 'hexagon', 'eyeglasses',
                  'bulldozer', 'harp', 'fish', 'line', 'pliers', 'speedboat', 'toaster', 'banana', 'purse', 'broccoli',
                  'toothpaste', 'dolphin', 'bottlecap', 'washing_machine', 'bird', 'pig', 'pillow', 'parrot', 'giraffe',
                  'lighthouse', 'cello', 'swan', 'donut', 'lantern', 'postcard', 'eye', 'finger', 'coffee_cup',
                  'aircraft_carrier', 'horse', 'paint_can', 'lollipop', 'snowman', 'skyscraper', 'stitches', 'lipstick',
                  'camouflage', 'book', 'rake', 'hot_air_balloon', 'saxophone', 'map', 'matches', 'tiger', 'bat',
                  'cannon', 'hockey_stick', 'power_outlet', 'screwdriver', 'marker', 'see_saw', 'barn', 'bucket',
                  'ladder', 'mouse', 'underwear', 'monkey', 'leg', 'train', 'mailbox', 'basketball', 'moustache',
                  'microwave', 'cactus', 'rainbow', 'fireplace', 'sheep', 'tent', 'fan', 'police_car', 'sandwich',
                  'sleeping_bag', 'guitar', 'mug', 'face', 'compass', 'spoon', 'broom', 'alarm_clock', 'snorkel',
                  'nose', 'hourglass', 'drill', 'cloud', 'cake', 'skull', 'saw', 'goatee', 'cup', 'dumbbell', 'apple',
                  'pizza', 'canoe', 'bicycle', 'grapes', 'toilet', 'blueberry', 'submarine', 'backpack', 'stethoscope',
                  'airplane', 'drums', 'fence', 'teapot', 'remote_control', 'calendar', 'suitcase', 'wine_bottle',
                  'helicopter', 'hot_dog', 'car', 'owl', 'mushroom', 'school_bus', 'whale', 'crocodile',
                  'roller_coaster', 'octopus', 'basket', 'potato', 'leaf', 'zigzag', 'syringe', 'pond', 'dragon',
                  'triangle', 'carrot', 'parachute', 'van', 'stove', 'bridge', 'swing_set', 'bush',
                  'The_Great_Wall_of_China', 'helmet', 'flying_saucer', 'trumpet', 'penguin', 'candle', 'peanut',
                  'beard', 'fire_hydrant', 'cow', 'lighter', 'waterslide', 'elephant', 'oven', 'flower', 'spider',
                  'motorbike', 'strawberry', 'diving_board', 'sweater', 'paper_clip', 'scissors', 'angel', 'key',
                  'yoga', 'fork', 'axe', 'rhinoceros', 'brain', 'flip_flops', 'elbow', 'light_bulb', 'radio', 'camel',
                  'paintbrush', 'watermelon', 'The_Mona_Lisa', 'stairs', 'picture_frame', 'clock', 'sun', 'bear',
                  'bench', 'camera', 'eraser', 'snail', 'beach', 'ocean', 'rifle', 'passport', 'ambulance', 'lobster',
                  'flamingo', 'streetlight', 'necklace'],
    'painting': ['raccoon', 'windmill', 'lightning', 'mouth', 'animal_migration', 'hammer', 'bathtub', 'television',
                 'birthday_cake', 'arm', 'telephone', 'firetruck', 'toothbrush', 'sailboat', 'river', 'baseball',
                 'scorpion', 'golf_club', 'ear', 'traffic_light', 'frying_pan', 'door', 'frog', 'keyboard', 'zebra',
                 'bread', 'umbrella', 'onion', 'stereo', 'dishwasher', 'pants', 'crown', 'hospital', 'square',
                 't-shirt', 'nail', 'squirrel', 'couch', 'hockey_puck', 'laptop', 'dog', 'steak', 'violin', 'cooler',
                 'bracelet', 'crab', 'cell_phone', 'shoe', 'ant', 'asparagus', 'garden_hose', 'rain', 'pineapple',
                 'table', 'rabbit', 'pickup_truck', 'peas', 'ice_cream', 'The_Eiffel_Tower', 'clarinet', 'string_bean',
                 'hedgehog', 'rollerskates', 'ceiling_fan', 'butterfly', 'flashlight', 'crayon', 'palm_tree', 'truck',
                 'shark', 'cat', 'binoculars', 'bee', 'star', 'foot', 'smiley_face', 'envelope', 'teddy-bear',
                 'feather', 'tooth', 'lion', 'toe', 'hand', 'boomerang', 'tree', 'sword', 'diamond', 'pool',
                 'hamburger', 'chandelier', 'floor_lamp', 'circle', 'kangaroo', 'hat', 'snake', 'house', 'sea_turtle',
                 'campfire', 'soccer_ball', 'megaphone', 'grass', 'jacket', 'mountain', 'cookie', 'wine_glass',
                 'octagon', 'church', 'cruise_ship', 'stop_sign', 'knife', 'belt', 'hurricane', 'piano', 'pear',
                 'wheel', 'castle', 'sink', 'headphones', 'bus', 'tennis_racquet', 'shovel', 'moon', 'hot_tub',
                 'trombone', 'pencil', 'duck', 'knee', 'blackberry', 'panda', 'garden', 'chair', 'tractor',
                 'house_plant', 'spreadsheet', 'skateboard', 'bandage', 'tornado', 'anvil', 'squiggle', 'mosquito',
                 'jail', 'snowflake', 'sock', 'bed', 'vase', 'baseball_bat', 'popsicle', 'bowtie', 'calculator',
                 'microphone', 'computer', 'wristwatch', 'shorts', 'dresser', 'mermaid', 'hexagon', 'eyeglasses',
                 'bulldozer', 'harp', 'fish', 'line', 'pliers', 'speedboat', 'toaster', 'banana', 'purse', 'broccoli',
                 'toothpaste', 'dolphin', 'bottlecap', 'washing_machine', 'bird', 'pig', 'pillow', 'parrot', 'giraffe',
                 'lighthouse', 'cello', 'swan', 'donut', 'lantern', 'postcard', 'eye', 'finger', 'coffee_cup',
                 'aircraft_carrier', 'horse', 'paint_can', 'lollipop', 'snowman', 'skyscraper', 'stitches', 'lipstick',
                 'camouflage', 'book', 'rake', 'hot_air_balloon', 'saxophone', 'map', 'matches', 'tiger', 'bat',
                 'cannon', 'hockey_stick', 'power_outlet', 'screwdriver', 'marker', 'see_saw', 'barn', 'bucket',
                 'ladder', 'mouse', 'underwear', 'monkey', 'leg', 'train', 'mailbox', 'basketball', 'moustache',
                 'microwave', 'cactus', 'rainbow', 'fireplace', 'sheep', 'tent', 'fan', 'police_car', 'sandwich',
                 'sleeping_bag', 'guitar', 'mug', 'face', 'compass', 'spoon', 'broom', 'alarm_clock', 'snorkel', 'nose',
                 'hourglass', 'drill', 'cloud', 'cake', 'skull', 'saw', 'goatee', 'cup', 'dumbbell', 'apple', 'pizza',
                 'canoe', 'bicycle', 'grapes', 'toilet', 'blueberry', 'submarine', 'backpack', 'stethoscope',
                 'airplane', 'drums', 'fence', 'teapot', 'remote_control', 'calendar', 'suitcase', 'wine_bottle',
                 'helicopter', 'hot_dog', 'car', 'owl', 'mushroom', 'school_bus', 'whale', 'crocodile',
                 'roller_coaster', 'octopus', 'basket', 'potato', 'leaf', 'zigzag', 'syringe', 'pond', 'dragon',
                 'triangle', 'carrot', 'parachute', 'van', 'stove', 'bridge', 'swing_set', 'bush',
                 'The_Great_Wall_of_China', 'helmet', 'flying_saucer', 'trumpet', 'penguin', 'candle', 'peanut',
                 'beard', 'fire_hydrant', 'cow', 'lighter', 'waterslide', 'elephant', 'oven', 'flower', 'spider',
                 'motorbike', 'strawberry', 'diving_board', 'sweater', 'paper_clip', 'scissors', 'angel', 'key', 'yoga',
                 'fork', 'axe', 'rhinoceros', 'brain', 'flip_flops', 'elbow', 'light_bulb', 'radio', 'camel',
                 'paintbrush', 'watermelon', 'The_Mona_Lisa', 'stairs', 'picture_frame', 'clock', 'sun', 'bear',
                 'bench', 'camera', 'eraser', 'snail', 'beach', 'ocean', 'rifle', 'passport', 'ambulance', 'lobster',
                 'flamingo', 'streetlight', 'necklace'],
    'real': ['raccoon', 'windmill', 'lightning', 'mouth', 'animal_migration', 'hammer', 'bathtub', 'television',
             'birthday_cake', 'arm', 'telephone', 'firetruck', 'toothbrush', 'sailboat', 'river', 'baseball',
             'scorpion', 'golf_club', 'ear', 'traffic_light', 'frying_pan', 'door', 'frog', 'keyboard', 'zebra',
             'bread', 'umbrella', 'onion', 'stereo', 'dishwasher', 'pants', 'crown', 'hospital', 'square', 't-shirt',
             'nail', 'squirrel', 'couch', 'hockey_puck', 'laptop', 'dog', 'steak', 'violin', 'cooler', 'bracelet',
             'crab', 'cell_phone', 'shoe', 'ant', 'asparagus', 'garden_hose', 'rain', 'pineapple', 'table', 'rabbit',
             'pickup_truck', 'peas', 'ice_cream', 'The_Eiffel_Tower', 'clarinet', 'string_bean', 'hedgehog',
             'rollerskates', 'ceiling_fan', 'butterfly', 'flashlight', 'crayon', 'palm_tree', 'truck', 'shark', 'cat',
             'binoculars', 'bee', 'star', 'foot', 'smiley_face', 'envelope', 'teddy-bear', 'feather', 'tooth', 'lion',
             'toe', 'hand', 'boomerang', 'tree', 'sword', 'diamond', 'pool', 'hamburger', 'chandelier', 'floor_lamp',
             'circle', 'kangaroo', 'hat', 'snake', 'house', 'sea_turtle', 'campfire', 'soccer_ball', 'megaphone',
             'grass', 'jacket', 'mountain', 'cookie', 'wine_glass', 'octagon', 'church', 'cruise_ship', 'stop_sign',
             'knife', 'belt', 'hurricane', 'piano', 'pear', 'wheel', 'castle', 'sink', 'headphones', 'bus',
             'tennis_racquet', 'shovel', 'moon', 'hot_tub', 'trombone', 'pencil', 'duck', 'knee', 'blackberry', 'panda',
             'garden', 'chair', 'tractor', 'house_plant', 'spreadsheet', 'skateboard', 'bandage', 'tornado', 'anvil',
             'squiggle', 'mosquito', 'jail', 'snowflake', 'sock', 'bed', 'vase', 'baseball_bat', 'popsicle', 'bowtie',
             'calculator', 'microphone', 'computer', 'wristwatch', 'shorts', 'dresser', 'mermaid', 'hexagon',
             'eyeglasses', 'bulldozer', 'harp', 'fish', 'line', 'pliers', 'speedboat', 'toaster', 'banana', 'purse',
             'broccoli', 'toothpaste', 'dolphin', 'bottlecap', 'washing_machine', 'bird', 'pig', 'pillow', 'parrot',
             'giraffe', 'lighthouse', 'cello', 'swan', 'donut', 'lantern', 'postcard', 'eye', 'finger', 'coffee_cup',
             'aircraft_carrier', 'horse', 'paint_can', 'lollipop', 'snowman', 'skyscraper', 'stitches', 'lipstick',
             'camouflage', 'book', 'rake', 'hot_air_balloon', 'saxophone', 'map', 'matches', 'tiger', 'bat', 'cannon',
             'hockey_stick', 'power_outlet', 'screwdriver', 'marker', 'see_saw', 'barn', 'bucket', 'ladder', 'mouse',
             'underwear', 'monkey', 'leg', 'train', 'mailbox', 'basketball', 'moustache', 'microwave', 'cactus',
             'rainbow', 'fireplace', 'sheep', 'tent', 'fan', 'police_car', 'sandwich', 'sleeping_bag', 'guitar', 'mug',
             'face', 'compass', 'spoon', 'broom', 'alarm_clock', 'snorkel', 'nose', 'hourglass', 'drill', 'cloud',
             'cake', 'skull', 'saw', 'goatee', 'cup', 'dumbbell', 'apple', 'pizza', 'canoe', 'bicycle', 'grapes',
             'toilet', 'blueberry', 'submarine', 'backpack', 'stethoscope', 'airplane', 'drums', 'fence', 'teapot',
             'remote_control', 'calendar', 'suitcase', 'wine_bottle', 'helicopter', 'hot_dog', 'car', 'owl', 'mushroom',
             'school_bus', 'whale', 'crocodile', 'roller_coaster', 'octopus', 'basket', 'potato', 'leaf', 'zigzag',
             'syringe', 'pond', 'dragon', 'triangle', 'carrot', 'parachute', 'van', 'stove', 'bridge', 'swing_set',
             'bush', 'The_Great_Wall_of_China', 'helmet', 'flying_saucer', 'trumpet', 'penguin', 'candle', 'peanut',
             'beard', 'fire_hydrant', 'cow', 'lighter', 'waterslide', 'elephant', 'oven', 'flower', 'spider',
             'motorbike', 'strawberry', 'diving_board', 'sweater', 'paper_clip', 'scissors', 'angel', 'key', 'yoga',
             'fork', 'axe', 'rhinoceros', 'brain', 'flip_flops', 'elbow', 'light_bulb', 'radio', 'camel', 'paintbrush',
             'watermelon', 'The_Mona_Lisa', 'stairs', 'picture_frame', 'clock', 'sun', 'bear', 'bench', 'camera',
             'eraser', 'snail', 'beach', 'ocean', 'rifle', 'passport', 'ambulance', 'lobster', 'flamingo',
             'streetlight', 'necklace'],
    'sketch': ['raccoon', 'windmill', 'lightning', 'mouth', 'animal_migration', 'hammer', 'bathtub', 'television',
               'birthday_cake', 'arm', 'telephone', 'firetruck', 'toothbrush', 'sailboat', 'river', 'baseball',
               'scorpion', 'golf_club', 'ear', 'traffic_light', 'frying_pan', 'door', 'frog', 'keyboard', 'zebra',
               'bread', 'umbrella', 'onion', 'stereo', 'dishwasher', 'pants', 'crown', 'hospital', 'square', 't-shirt',
               'nail', 'squirrel', 'couch', 'hockey_puck', 'laptop', 'dog', 'steak', 'violin', 'cooler', 'bracelet',
               'crab', 'cell_phone', 'shoe', 'ant', 'asparagus', 'garden_hose', 'rain', 'pineapple', 'table', 'rabbit',
               'pickup_truck', 'peas', 'ice_cream', 'The_Eiffel_Tower', 'clarinet', 'string_bean', 'hedgehog',
               'rollerskates', 'ceiling_fan', 'butterfly', 'flashlight', 'crayon', 'palm_tree', 'truck', 'shark', 'cat',
               'binoculars', 'bee', 'star', 'foot', 'smiley_face', 'envelope', 'teddy-bear', 'feather', 'tooth', 'lion',
               'toe', 'hand', 'boomerang', 'tree', 'sword', 'diamond', 'pool', 'hamburger', 'chandelier', 'floor_lamp',
               'circle', 'kangaroo', 'hat', 'snake', 'house', 'sea_turtle', 'campfire', 'soccer_ball', 'megaphone',
               'grass', 'jacket', 'mountain', 'cookie', 'wine_glass', 'octagon', 'church', 'cruise_ship', 'stop_sign',
               'knife', 'belt', 'hurricane', 'piano', 'pear', 'wheel', 'castle', 'sink', 'headphones', 'bus',
               'tennis_racquet', 'shovel', 'moon', 'hot_tub', 'trombone', 'pencil', 'duck', 'knee', 'blackberry',
               'panda', 'garden', 'chair', 'tractor', 'house_plant', 'spreadsheet', 'skateboard', 'bandage', 'tornado',
               'anvil', 'squiggle', 'mosquito', 'jail', 'snowflake', 'sock', 'bed', 'vase', 'baseball_bat', 'popsicle',
               'bowtie', 'calculator', 'microphone', 'computer', 'wristwatch', 'shorts', 'dresser', 'mermaid',
               'hexagon', 'eyeglasses', 'bulldozer', 'harp', 'fish', 'line', 'pliers', 'speedboat', 'toaster', 'banana',
               'purse', 'broccoli', 'toothpaste', 'dolphin', 'bottlecap', 'washing_machine', 'bird', 'pig', 'pillow',
               'parrot', 'giraffe', 'lighthouse', 'cello', 'swan', 'donut', 'lantern', 'postcard', 'eye', 'finger',
               'coffee_cup', 'aircraft_carrier', 'horse', 'paint_can', 'lollipop', 'snowman', 'skyscraper', 'stitches',
               'lipstick', 'camouflage', 'book', 'rake', 'hot_air_balloon', 'saxophone', 'map', 'matches', 'tiger',
               'bat', 'cannon', 'hockey_stick', 'power_outlet', 'screwdriver', 'marker', 'see_saw', 'barn', 'bucket',
               'ladder', 'mouse', 'underwear', 'monkey', 'leg', 'train', 'mailbox', 'basketball', 'moustache',
               'microwave', 'cactus', 'rainbow', 'fireplace', 'sheep', 'tent', 'fan', 'police_car', 'sandwich',
               'sleeping_bag', 'guitar', 'mug', 'face', 'compass', 'spoon', 'broom', 'alarm_clock', 'snorkel', 'nose',
               'hourglass', 'drill', 'cloud', 'cake', 'skull', 'saw', 'goatee', 'cup', 'dumbbell', 'apple', 'pizza',
               'canoe', 'bicycle', 'grapes', 'toilet', 'blueberry', 'submarine', 'backpack', 'stethoscope', 'airplane',
               'drums', 'fence', 'teapot', 'remote_control', 'calendar', 'suitcase', 'wine_bottle', 'helicopter',
               'hot_dog', 'car', 'owl', 'mushroom', 'school_bus', 'whale', 'crocodile', 'roller_coaster', 'octopus',
               'basket', 'potato', 'leaf', 'zigzag', 'syringe', 'pond', 'dragon', 'triangle', 'carrot', 'parachute',
               'van', 'stove', 'bridge', 'swing_set', 'bush', 'The_Great_Wall_of_China', 'helmet', 'flying_saucer',
               'trumpet', 'penguin', 'candle', 'peanut', 'beard', 'fire_hydrant', 'cow', 'lighter', 'waterslide',
               'elephant', 'oven', 'flower', 'spider', 'motorbike', 'strawberry', 'diving_board', 'sweater',
               'paper_clip', 'scissors', 'angel', 'key', 'yoga', 'fork', 'axe', 'rhinoceros', 'brain', 'flip_flops',
               'elbow', 'light_bulb', 'radio', 'camel', 'paintbrush', 'watermelon', 'The_Mona_Lisa', 'stairs',
               'picture_frame', 'clock', 'sun', 'bear', 'bench', 'camera', 'eraser', 'snail', 'beach', 'ocean', 'rifle',
               'passport', 'ambulance', 'lobster', 'flamingo', 'streetlight', 'necklace']
    }

data_settings['domain-net']['CPQRS_I']['C'] = {
    'clipart': ['raccoon', 'windmill', 'lightning', 'mouth', 'animal_migration', 'hammer', 'bathtub', 'television',
                'birthday_cake', 'arm', 'telephone', 'firetruck', 'toothbrush', 'sailboat', 'river', 'baseball',
                'scorpion', 'golf_club', 'ear', 'traffic_light', 'frying_pan', 'door', 'frog', 'keyboard', 'zebra',
                'bread', 'umbrella', 'onion', 'stereo', 'dishwasher', 'pants', 'crown', 'hospital', 'square', 't-shirt',
                'nail', 'squirrel', 'couch', 'hockey_puck', 'laptop', 'dog', 'steak', 'violin', 'cooler', 'bracelet',
                'crab', 'cell_phone', 'shoe', 'ant', 'asparagus', 'garden_hose', 'rain', 'pineapple', 'table', 'rabbit',
                'pickup_truck', 'peas', 'ice_cream', 'The_Eiffel_Tower', 'clarinet', 'string_bean', 'hedgehog',
                'rollerskates', 'ceiling_fan', 'butterfly', 'flashlight', 'crayon', 'palm_tree', 'truck', 'shark',
                'cat', 'binoculars', 'bee', 'star', 'foot', 'smiley_face', 'envelope', 'teddy-bear', 'feather', 'tooth',
                'lion', 'toe', 'hand', 'boomerang', 'tree', 'sword', 'diamond', 'pool', 'hamburger', 'chandelier',
                'floor_lamp', 'circle', 'kangaroo', 'hat', 'snake', 'house', 'sea_turtle', 'campfire', 'soccer_ball',
                'megaphone', 'grass', 'jacket', 'mountain', 'cookie', 'wine_glass', 'octagon', 'church', 'cruise_ship',
                'stop_sign', 'knife', 'belt', 'hurricane', 'piano', 'pear', 'wheel', 'castle', 'sink', 'headphones',
                'bus', 'tennis_racquet', 'shovel', 'moon', 'hot_tub', 'trombone', 'pencil', 'duck', 'knee',
                'blackberry', 'panda', 'garden', 'chair', 'tractor', 'house_plant', 'spreadsheet', 'skateboard',
                'bandage', 'tornado', 'anvil', 'squiggle', 'mosquito', 'jail', 'snowflake', 'sock', 'bed', 'vase',
                'baseball_bat', 'popsicle', 'bowtie', 'calculator', 'microphone', 'computer', 'wristwatch', 'shorts',
                'dresser', 'mermaid', 'hexagon', 'eyeglasses', 'bulldozer', 'harp', 'fish', 'line', 'pliers',
                'speedboat', 'toaster', 'banana', 'purse', 'broccoli', 'toothpaste', 'dolphin', 'bottlecap',
                'washing_machine', 'bird', 'pig', 'pillow', 'parrot', 'giraffe', 'lighthouse', 'cello', 'swan', 'donut',
                'lantern', 'postcard', 'eye', 'finger', 'coffee_cup', 'aircraft_carrier', 'horse', 'paint_can',
                'lollipop', 'snowman', 'skyscraper', 'stitches', 'lipstick', 'camouflage', 'book', 'rake',
                'hot_air_balloon', 'saxophone', 'map', 'matches', 'tiger', 'bat', 'cannon', 'hockey_stick',
                'power_outlet', 'screwdriver', 'marker', 'see_saw', 'barn', 'bucket', 'ladder', 'mouse', 'underwear',
                'monkey', 'leg', 'train', 'mailbox', 'basketball', 'moustache', 'microwave', 'cactus', 'rainbow',
                'fireplace', 'sheep', 'tent', 'fan', 'police_car', 'sandwich', 'sleeping_bag', 'guitar', 'mug', 'face',
                'compass', 'spoon', 'broom', 'alarm_clock', 'snorkel', 'nose', 'hourglass', 'drill', 'cloud', 'cake',
                'skull', 'saw', 'goatee', 'cup', 'dumbbell', 'apple', 'pizza', 'canoe', 'bicycle', 'grapes', 'toilet',
                'blueberry', 'submarine', 'backpack', 'stethoscope', 'airplane', 'drums', 'fence', 'teapot',
                'remote_control', 'calendar', 'suitcase', 'wine_bottle', 'helicopter', 'hot_dog', 'car', 'owl',
                'mushroom', 'school_bus', 'whale', 'crocodile', 'roller_coaster', 'octopus', 'basket', 'potato', 'leaf',
                'zigzag', 'syringe', 'pond', 'dragon', 'triangle', 'carrot', 'parachute', 'van', 'stove', 'bridge',
                'swing_set', 'bush', 'The_Great_Wall_of_China', 'helmet', 'flying_saucer', 'trumpet', 'penguin',
                'candle', 'peanut', 'beard', 'fire_hydrant', 'cow', 'lighter', 'waterslide', 'elephant', 'oven',
                'flower', 'spider', 'motorbike', 'strawberry', 'diving_board', 'sweater', 'paper_clip', 'scissors',
                'angel', 'key', 'yoga', 'fork', 'axe', 'rhinoceros', 'brain', 'flip_flops', 'elbow', 'light_bulb',
                'radio', 'camel', 'paintbrush', 'watermelon', 'The_Mona_Lisa', 'stairs', 'picture_frame', 'clock',
                'sun', 'bear', 'bench', 'camera', 'eraser', 'snail', 'beach', 'ocean', 'rifle', 'passport', 'ambulance',
                'lobster', 'flamingo', 'streetlight', 'necklace'],
    'painting': ['raccoon', 'windmill', 'lightning', 'mouth', 'animal_migration', 'hammer', 'bathtub', 'television',
                 'birthday_cake', 'arm', 'telephone', 'firetruck', 'toothbrush', 'sailboat', 'river', 'baseball',
                 'scorpion', 'golf_club', 'ear', 'traffic_light', 'frying_pan', 'door', 'frog', 'keyboard', 'zebra',
                 'bread', 'umbrella', 'onion', 'stereo', 'dishwasher', 'pants', 'crown', 'hospital', 'square',
                 't-shirt', 'nail', 'squirrel', 'couch', 'hockey_puck', 'laptop', 'dog', 'steak', 'violin', 'cooler',
                 'bracelet', 'crab', 'cell_phone', 'shoe', 'ant', 'asparagus', 'garden_hose', 'rain', 'pineapple',
                 'table', 'rabbit', 'pickup_truck', 'peas', 'ice_cream', 'The_Eiffel_Tower', 'clarinet', 'string_bean',
                 'hedgehog', 'rollerskates', 'ceiling_fan', 'butterfly', 'flashlight', 'crayon', 'palm_tree', 'truck',
                 'shark', 'cat', 'binoculars', 'bee', 'star', 'foot', 'smiley_face', 'envelope', 'teddy-bear',
                 'feather', 'tooth', 'lion', 'toe', 'hand', 'boomerang', 'tree', 'sword', 'diamond', 'pool',
                 'hamburger', 'chandelier', 'floor_lamp', 'circle', 'kangaroo', 'hat', 'snake', 'house', 'sea_turtle',
                 'campfire', 'soccer_ball', 'megaphone', 'grass', 'jacket', 'mountain', 'cookie', 'wine_glass',
                 'octagon', 'church', 'cruise_ship', 'stop_sign', 'knife', 'belt', 'hurricane', 'piano', 'pear',
                 'wheel', 'castle', 'sink', 'headphones', 'bus', 'tennis_racquet', 'shovel', 'moon', 'hot_tub',
                 'trombone', 'pencil', 'duck', 'knee', 'blackberry', 'panda', 'garden', 'chair', 'tractor',
                 'house_plant', 'spreadsheet', 'skateboard', 'bandage', 'tornado', 'anvil', 'squiggle', 'mosquito',
                 'jail', 'snowflake', 'sock', 'bed', 'vase', 'baseball_bat', 'popsicle', 'bowtie', 'calculator',
                 'microphone', 'computer', 'wristwatch', 'shorts', 'dresser', 'mermaid', 'hexagon', 'eyeglasses',
                 'bulldozer', 'harp', 'fish', 'line', 'pliers', 'speedboat', 'toaster', 'banana', 'purse', 'broccoli',
                 'toothpaste', 'dolphin', 'bottlecap', 'washing_machine', 'bird', 'pig', 'pillow', 'parrot', 'giraffe',
                 'lighthouse', 'cello', 'swan', 'donut', 'lantern', 'postcard', 'eye', 'finger', 'coffee_cup',
                 'aircraft_carrier', 'horse', 'paint_can', 'lollipop', 'snowman', 'skyscraper', 'stitches', 'lipstick',
                 'camouflage', 'book', 'rake', 'hot_air_balloon', 'saxophone', 'map', 'matches', 'tiger', 'bat',
                 'cannon', 'hockey_stick', 'power_outlet', 'screwdriver', 'marker', 'see_saw', 'barn', 'bucket',
                 'ladder', 'mouse', 'underwear', 'monkey', 'leg', 'train', 'mailbox', 'basketball', 'moustache',
                 'microwave', 'cactus', 'rainbow', 'fireplace', 'sheep', 'tent', 'fan', 'police_car', 'sandwich',
                 'sleeping_bag', 'guitar', 'mug', 'face', 'compass', 'spoon', 'broom', 'alarm_clock', 'snorkel', 'nose',
                 'hourglass', 'drill', 'cloud', 'cake', 'skull', 'saw', 'goatee', 'cup', 'dumbbell', 'apple', 'pizza',
                 'canoe', 'bicycle', 'grapes', 'toilet', 'blueberry', 'submarine', 'backpack', 'stethoscope',
                 'airplane', 'drums', 'fence', 'teapot', 'remote_control', 'calendar', 'suitcase', 'wine_bottle',
                 'helicopter', 'hot_dog', 'car', 'owl', 'mushroom', 'school_bus', 'whale', 'crocodile',
                 'roller_coaster', 'octopus', 'basket', 'potato', 'leaf', 'zigzag', 'syringe', 'pond', 'dragon',
                 'triangle', 'carrot', 'parachute', 'van', 'stove', 'bridge', 'swing_set', 'bush',
                 'The_Great_Wall_of_China', 'helmet', 'flying_saucer', 'trumpet', 'penguin', 'candle', 'peanut',
                 'beard', 'fire_hydrant', 'cow', 'lighter', 'waterslide', 'elephant', 'oven', 'flower', 'spider',
                 'motorbike', 'strawberry', 'diving_board', 'sweater', 'paper_clip', 'scissors', 'angel', 'key', 'yoga',
                 'fork', 'axe', 'rhinoceros', 'brain', 'flip_flops', 'elbow', 'light_bulb', 'radio', 'camel',
                 'paintbrush', 'watermelon', 'The_Mona_Lisa', 'stairs', 'picture_frame', 'clock', 'sun', 'bear',
                 'bench', 'camera', 'eraser', 'snail', 'beach', 'ocean', 'rifle', 'passport', 'ambulance', 'lobster',
                 'flamingo', 'streetlight', 'necklace'],
    'quickdraw': ['raccoon', 'windmill', 'lightning', 'mouth', 'animal_migration', 'hammer', 'bathtub', 'television',
                  'birthday_cake', 'arm', 'telephone', 'firetruck', 'toothbrush', 'sailboat', 'river', 'baseball',
                  'scorpion', 'golf_club', 'ear', 'traffic_light', 'frying_pan', 'door', 'frog', 'keyboard', 'zebra',
                  'bread', 'umbrella', 'onion', 'stereo', 'dishwasher', 'pants', 'crown', 'hospital', 'square',
                  't-shirt', 'nail', 'squirrel', 'couch', 'hockey_puck', 'laptop', 'dog', 'steak', 'violin', 'cooler',
                  'bracelet', 'crab', 'cell_phone', 'shoe', 'ant', 'asparagus', 'garden_hose', 'rain', 'pineapple',
                  'table', 'rabbit', 'pickup_truck', 'peas', 'ice_cream', 'The_Eiffel_Tower', 'clarinet', 'string_bean',
                  'hedgehog', 'rollerskates', 'ceiling_fan', 'butterfly', 'flashlight', 'crayon', 'palm_tree', 'truck',
                  'shark', 'cat', 'binoculars', 'bee', 'star', 'foot', 'smiley_face', 'envelope', 'teddy-bear',
                  'feather', 'tooth', 'lion', 'toe', 'hand', 'boomerang', 'tree', 'sword', 'diamond', 'pool',
                  'hamburger', 'chandelier', 'floor_lamp', 'circle', 'kangaroo', 'hat', 'snake', 'house', 'sea_turtle',
                  'campfire', 'soccer_ball', 'megaphone', 'grass', 'jacket', 'mountain', 'cookie', 'wine_glass',
                  'octagon', 'church', 'cruise_ship', 'stop_sign', 'knife', 'belt', 'hurricane', 'piano', 'pear',
                  'wheel', 'castle', 'sink', 'headphones', 'bus', 'tennis_racquet', 'shovel', 'moon', 'hot_tub',
                  'trombone', 'pencil', 'duck', 'knee', 'blackberry', 'panda', 'garden', 'chair', 'tractor',
                  'house_plant', 'spreadsheet', 'skateboard', 'bandage', 'tornado', 'anvil', 'squiggle', 'mosquito',
                  'jail', 'snowflake', 'sock', 'bed', 'vase', 'baseball_bat', 'popsicle', 'bowtie', 'calculator',
                  'microphone', 'computer', 'wristwatch', 'shorts', 'dresser', 'mermaid', 'hexagon', 'eyeglasses',
                  'bulldozer', 'harp', 'fish', 'line', 'pliers', 'speedboat', 'toaster', 'banana', 'purse', 'broccoli',
                  'toothpaste', 'dolphin', 'bottlecap', 'washing_machine', 'bird', 'pig', 'pillow', 'parrot', 'giraffe',
                  'lighthouse', 'cello', 'swan', 'donut', 'lantern', 'postcard', 'eye', 'finger', 'coffee_cup',
                  'aircraft_carrier', 'horse', 'paint_can', 'lollipop', 'snowman', 'skyscraper', 'stitches', 'lipstick',
                  'camouflage', 'book', 'rake', 'hot_air_balloon', 'saxophone', 'map', 'matches', 'tiger', 'bat',
                  'cannon', 'hockey_stick', 'power_outlet', 'screwdriver', 'marker', 'see_saw', 'barn', 'bucket',
                  'ladder', 'mouse', 'underwear', 'monkey', 'leg', 'train', 'mailbox', 'basketball', 'moustache',
                  'microwave', 'cactus', 'rainbow', 'fireplace', 'sheep', 'tent', 'fan', 'police_car', 'sandwich',
                  'sleeping_bag', 'guitar', 'mug', 'face', 'compass', 'spoon', 'broom', 'alarm_clock', 'snorkel',
                  'nose', 'hourglass', 'drill', 'cloud', 'cake', 'skull', 'saw', 'goatee', 'cup', 'dumbbell', 'apple',
                  'pizza', 'canoe', 'bicycle', 'grapes', 'toilet', 'blueberry', 'submarine', 'backpack', 'stethoscope',
                  'airplane', 'drums', 'fence', 'teapot', 'remote_control', 'calendar', 'suitcase', 'wine_bottle',
                  'helicopter', 'hot_dog', 'car', 'owl', 'mushroom', 'school_bus', 'whale', 'crocodile',
                  'roller_coaster', 'octopus', 'basket', 'potato', 'leaf', 'zigzag', 'syringe', 'pond', 'dragon',
                  'triangle', 'carrot', 'parachute', 'van', 'stove', 'bridge', 'swing_set', 'bush',
                  'The_Great_Wall_of_China', 'helmet', 'flying_saucer', 'trumpet', 'penguin', 'candle', 'peanut',
                  'beard', 'fire_hydrant', 'cow', 'lighter', 'waterslide', 'elephant', 'oven', 'flower', 'spider',
                  'motorbike', 'strawberry', 'diving_board', 'sweater', 'paper_clip', 'scissors', 'angel', 'key',
                  'yoga', 'fork', 'axe', 'rhinoceros', 'brain', 'flip_flops', 'elbow', 'light_bulb', 'radio', 'camel',
                  'paintbrush', 'watermelon', 'The_Mona_Lisa', 'stairs', 'picture_frame', 'clock', 'sun', 'bear',
                  'bench', 'camera', 'eraser', 'snail', 'beach', 'ocean', 'rifle', 'passport', 'ambulance', 'lobster',
                  'flamingo', 'streetlight', 'necklace'],
    'real': ['raccoon', 'windmill', 'lightning', 'mouth', 'animal_migration', 'hammer', 'bathtub', 'television',
             'birthday_cake', 'arm', 'telephone', 'firetruck', 'toothbrush', 'sailboat', 'river', 'baseball',
             'scorpion', 'golf_club', 'ear', 'traffic_light', 'frying_pan', 'door', 'frog', 'keyboard', 'zebra',
             'bread', 'umbrella', 'onion', 'stereo', 'dishwasher', 'pants', 'crown', 'hospital', 'square', 't-shirt',
             'nail', 'squirrel', 'couch', 'hockey_puck', 'laptop', 'dog', 'steak', 'violin', 'cooler', 'bracelet',
             'crab', 'cell_phone', 'shoe', 'ant', 'asparagus', 'garden_hose', 'rain', 'pineapple', 'table', 'rabbit',
             'pickup_truck', 'peas', 'ice_cream', 'The_Eiffel_Tower', 'clarinet', 'string_bean', 'hedgehog',
             'rollerskates', 'ceiling_fan', 'butterfly', 'flashlight', 'crayon', 'palm_tree', 'truck', 'shark', 'cat',
             'binoculars', 'bee', 'star', 'foot', 'smiley_face', 'envelope', 'teddy-bear', 'feather', 'tooth', 'lion',
             'toe', 'hand', 'boomerang', 'tree', 'sword', 'diamond', 'pool', 'hamburger', 'chandelier', 'floor_lamp',
             'circle', 'kangaroo', 'hat', 'snake', 'house', 'sea_turtle', 'campfire', 'soccer_ball', 'megaphone',
             'grass', 'jacket', 'mountain', 'cookie', 'wine_glass', 'octagon', 'church', 'cruise_ship', 'stop_sign',
             'knife', 'belt', 'hurricane', 'piano', 'pear', 'wheel', 'castle', 'sink', 'headphones', 'bus',
             'tennis_racquet', 'shovel', 'moon', 'hot_tub', 'trombone', 'pencil', 'duck', 'knee', 'blackberry', 'panda',
             'garden', 'chair', 'tractor', 'house_plant', 'spreadsheet', 'skateboard', 'bandage', 'tornado', 'anvil',
             'squiggle', 'mosquito', 'jail', 'snowflake', 'sock', 'bed', 'vase', 'baseball_bat', 'popsicle', 'bowtie',
             'calculator', 'microphone', 'computer', 'wristwatch', 'shorts', 'dresser', 'mermaid', 'hexagon',
             'eyeglasses', 'bulldozer', 'harp', 'fish', 'line', 'pliers', 'speedboat', 'toaster', 'banana', 'purse',
             'broccoli', 'toothpaste', 'dolphin', 'bottlecap', 'washing_machine', 'bird', 'pig', 'pillow', 'parrot',
             'giraffe', 'lighthouse', 'cello', 'swan', 'donut', 'lantern', 'postcard', 'eye', 'finger', 'coffee_cup',
             'aircraft_carrier', 'horse', 'paint_can', 'lollipop', 'snowman', 'skyscraper', 'stitches', 'lipstick',
             'camouflage', 'book', 'rake', 'hot_air_balloon', 'saxophone', 'map', 'matches', 'tiger', 'bat', 'cannon',
             'hockey_stick', 'power_outlet', 'screwdriver', 'marker', 'see_saw', 'barn', 'bucket', 'ladder', 'mouse',
             'underwear', 'monkey', 'leg', 'train', 'mailbox', 'basketball', 'moustache', 'microwave', 'cactus',
             'rainbow', 'fireplace', 'sheep', 'tent', 'fan', 'police_car', 'sandwich', 'sleeping_bag', 'guitar', 'mug',
             'face', 'compass', 'spoon', 'broom', 'alarm_clock', 'snorkel', 'nose', 'hourglass', 'drill', 'cloud',
             'cake', 'skull', 'saw', 'goatee', 'cup', 'dumbbell', 'apple', 'pizza', 'canoe', 'bicycle', 'grapes',
             'toilet', 'blueberry', 'submarine', 'backpack', 'stethoscope', 'airplane', 'drums', 'fence', 'teapot',
             'remote_control', 'calendar', 'suitcase', 'wine_bottle', 'helicopter', 'hot_dog', 'car', 'owl', 'mushroom',
             'school_bus', 'whale', 'crocodile', 'roller_coaster', 'octopus', 'basket', 'potato', 'leaf', 'zigzag',
             'syringe', 'pond', 'dragon', 'triangle', 'carrot', 'parachute', 'van', 'stove', 'bridge', 'swing_set',
             'bush', 'The_Great_Wall_of_China', 'helmet', 'flying_saucer', 'trumpet', 'penguin', 'candle', 'peanut',
             'beard', 'fire_hydrant', 'cow', 'lighter', 'waterslide', 'elephant', 'oven', 'flower', 'spider',
             'motorbike', 'strawberry', 'diving_board', 'sweater', 'paper_clip', 'scissors', 'angel', 'key', 'yoga',
             'fork', 'axe', 'rhinoceros', 'brain', 'flip_flops', 'elbow', 'light_bulb', 'radio', 'camel', 'paintbrush',
             'watermelon', 'The_Mona_Lisa', 'stairs', 'picture_frame', 'clock', 'sun', 'bear', 'bench', 'camera',
             'eraser', 'snail', 'beach', 'ocean', 'rifle', 'passport', 'ambulance', 'lobster', 'flamingo',
             'streetlight', 'necklace'],
    'sketch': ['raccoon', 'windmill', 'lightning', 'mouth', 'animal_migration', 'hammer', 'bathtub', 'television',
               'birthday_cake', 'arm', 'telephone', 'firetruck', 'toothbrush', 'sailboat', 'river', 'baseball',
               'scorpion', 'golf_club', 'ear', 'traffic_light', 'frying_pan', 'door', 'frog', 'keyboard', 'zebra',
               'bread', 'umbrella', 'onion', 'stereo', 'dishwasher', 'pants', 'crown', 'hospital', 'square', 't-shirt',
               'nail', 'squirrel', 'couch', 'hockey_puck', 'laptop', 'dog', 'steak', 'violin', 'cooler', 'bracelet',
               'crab', 'cell_phone', 'shoe', 'ant', 'asparagus', 'garden_hose', 'rain', 'pineapple', 'table', 'rabbit',
               'pickup_truck', 'peas', 'ice_cream', 'The_Eiffel_Tower', 'clarinet', 'string_bean', 'hedgehog',
               'rollerskates', 'ceiling_fan', 'butterfly', 'flashlight', 'crayon', 'palm_tree', 'truck', 'shark', 'cat',
               'binoculars', 'bee', 'star', 'foot', 'smiley_face', 'envelope', 'teddy-bear', 'feather', 'tooth', 'lion',
               'toe', 'hand', 'boomerang', 'tree', 'sword', 'diamond', 'pool', 'hamburger', 'chandelier', 'floor_lamp',
               'circle', 'kangaroo', 'hat', 'snake', 'house', 'sea_turtle', 'campfire', 'soccer_ball', 'megaphone',
               'grass', 'jacket', 'mountain', 'cookie', 'wine_glass', 'octagon', 'church', 'cruise_ship', 'stop_sign',
               'knife', 'belt', 'hurricane', 'piano', 'pear', 'wheel', 'castle', 'sink', 'headphones', 'bus',
               'tennis_racquet', 'shovel', 'moon', 'hot_tub', 'trombone', 'pencil', 'duck', 'knee', 'blackberry',
               'panda', 'garden', 'chair', 'tractor', 'house_plant', 'spreadsheet', 'skateboard', 'bandage', 'tornado',
               'anvil', 'squiggle', 'mosquito', 'jail', 'snowflake', 'sock', 'bed', 'vase', 'baseball_bat', 'popsicle',
               'bowtie', 'calculator', 'microphone', 'computer', 'wristwatch', 'shorts', 'dresser', 'mermaid',
               'hexagon', 'eyeglasses', 'bulldozer', 'harp', 'fish', 'line', 'pliers', 'speedboat', 'toaster', 'banana',
               'purse', 'broccoli', 'toothpaste', 'dolphin', 'bottlecap', 'washing_machine', 'bird', 'pig', 'pillow',
               'parrot', 'giraffe', 'lighthouse', 'cello', 'swan', 'donut', 'lantern', 'postcard', 'eye', 'finger',
               'coffee_cup', 'aircraft_carrier', 'horse', 'paint_can', 'lollipop', 'snowman', 'skyscraper', 'stitches',
               'lipstick', 'camouflage', 'book', 'rake', 'hot_air_balloon', 'saxophone', 'map', 'matches', 'tiger',
               'bat', 'cannon', 'hockey_stick', 'power_outlet', 'screwdriver', 'marker', 'see_saw', 'barn', 'bucket',
               'ladder', 'mouse', 'underwear', 'monkey', 'leg', 'train', 'mailbox', 'basketball', 'moustache',
               'microwave', 'cactus', 'rainbow', 'fireplace', 'sheep', 'tent', 'fan', 'police_car', 'sandwich',
               'sleeping_bag', 'guitar', 'mug', 'face', 'compass', 'spoon', 'broom', 'alarm_clock', 'snorkel', 'nose',
               'hourglass', 'drill', 'cloud', 'cake', 'skull', 'saw', 'goatee', 'cup', 'dumbbell', 'apple', 'pizza',
               'canoe', 'bicycle', 'grapes', 'toilet', 'blueberry', 'submarine', 'backpack', 'stethoscope', 'airplane',
               'drums', 'fence', 'teapot', 'remote_control', 'calendar', 'suitcase', 'wine_bottle', 'helicopter',
               'hot_dog', 'car', 'owl', 'mushroom', 'school_bus', 'whale', 'crocodile', 'roller_coaster', 'octopus',
               'basket', 'potato', 'leaf', 'zigzag', 'syringe', 'pond', 'dragon', 'triangle', 'carrot', 'parachute',
               'van', 'stove', 'bridge', 'swing_set', 'bush', 'The_Great_Wall_of_China', 'helmet', 'flying_saucer',
               'trumpet', 'penguin', 'candle', 'peanut', 'beard', 'fire_hydrant', 'cow', 'lighter', 'waterslide',
               'elephant', 'oven', 'flower', 'spider', 'motorbike', 'strawberry', 'diving_board', 'sweater',
               'paper_clip', 'scissors', 'angel', 'key', 'yoga', 'fork', 'axe', 'rhinoceros', 'brain', 'flip_flops',
               'elbow', 'light_bulb', 'radio', 'camel', 'paintbrush', 'watermelon', 'The_Mona_Lisa', 'stairs',
               'picture_frame', 'clock', 'sun', 'bear', 'bench', 'camera', 'eraser', 'snail', 'beach', 'ocean', 'rifle',
               'passport', 'ambulance', 'lobster', 'flamingo', 'streetlight', 'necklace']
    }

data_settings['domain-net']['CIPQR_S']['C'] = {
    'clipart': ['raccoon', 'windmill', 'lightning', 'mouth', 'animal_migration', 'hammer', 'bathtub', 'television',
                'birthday_cake', 'arm', 'telephone', 'firetruck', 'toothbrush', 'sailboat', 'river', 'baseball',
                'scorpion', 'golf_club', 'ear', 'traffic_light', 'frying_pan', 'door', 'frog', 'keyboard', 'zebra',
                'bread', 'umbrella', 'onion', 'stereo', 'dishwasher', 'pants', 'crown', 'hospital', 'square', 't-shirt',
                'nail', 'squirrel', 'couch', 'hockey_puck', 'laptop', 'dog', 'steak', 'violin', 'cooler', 'bracelet',
                'crab', 'cell_phone', 'shoe', 'ant', 'asparagus', 'garden_hose', 'rain', 'pineapple', 'table', 'rabbit',
                'pickup_truck', 'peas', 'ice_cream', 'The_Eiffel_Tower', 'clarinet', 'string_bean', 'hedgehog',
                'rollerskates', 'ceiling_fan', 'butterfly', 'flashlight', 'crayon', 'palm_tree', 'truck', 'shark',
                'cat', 'binoculars', 'bee', 'star', 'foot', 'smiley_face', 'envelope', 'teddy-bear', 'feather', 'tooth',
                'lion', 'toe', 'hand', 'boomerang', 'tree', 'sword', 'diamond', 'pool', 'hamburger', 'chandelier',
                'floor_lamp', 'circle', 'kangaroo', 'hat', 'snake', 'house', 'sea_turtle', 'campfire', 'soccer_ball',
                'megaphone', 'grass', 'jacket', 'mountain', 'cookie', 'wine_glass', 'octagon', 'church', 'cruise_ship',
                'stop_sign', 'knife', 'belt', 'hurricane', 'piano', 'pear', 'wheel', 'castle', 'sink', 'headphones',
                'bus', 'tennis_racquet', 'shovel', 'moon', 'hot_tub', 'trombone', 'pencil', 'duck', 'knee',
                'blackberry', 'panda', 'garden', 'chair', 'tractor', 'house_plant', 'spreadsheet', 'skateboard',
                'bandage', 'tornado', 'anvil', 'squiggle', 'mosquito', 'jail', 'snowflake', 'sock', 'bed', 'vase',
                'baseball_bat', 'popsicle', 'bowtie', 'calculator', 'microphone', 'computer', 'wristwatch', 'shorts',
                'dresser', 'mermaid', 'hexagon', 'eyeglasses', 'bulldozer', 'harp', 'fish', 'line', 'pliers',
                'speedboat', 'toaster', 'banana', 'purse', 'broccoli', 'toothpaste', 'dolphin', 'bottlecap',
                'washing_machine', 'bird', 'pig', 'pillow', 'parrot', 'giraffe', 'lighthouse', 'cello', 'swan', 'donut',
                'lantern', 'postcard', 'eye', 'finger', 'coffee_cup', 'aircraft_carrier', 'horse', 'paint_can',
                'lollipop', 'snowman', 'skyscraper', 'stitches', 'lipstick', 'camouflage', 'book', 'rake',
                'hot_air_balloon', 'saxophone', 'map', 'matches', 'tiger', 'bat', 'cannon', 'hockey_stick',
                'power_outlet', 'screwdriver', 'marker', 'see_saw', 'barn', 'bucket', 'ladder', 'mouse', 'underwear',
                'monkey', 'leg', 'train', 'mailbox', 'basketball', 'moustache', 'microwave', 'cactus', 'rainbow',
                'fireplace', 'sheep', 'tent', 'fan', 'police_car', 'sandwich', 'sleeping_bag', 'guitar', 'mug', 'face',
                'compass', 'spoon', 'broom', 'alarm_clock', 'snorkel', 'nose', 'hourglass', 'drill', 'cloud', 'cake',
                'skull', 'saw', 'goatee', 'cup', 'dumbbell', 'apple', 'pizza', 'canoe', 'bicycle', 'grapes', 'toilet',
                'blueberry', 'submarine', 'backpack', 'stethoscope', 'airplane', 'drums', 'fence', 'teapot',
                'remote_control', 'calendar', 'suitcase', 'wine_bottle', 'helicopter', 'hot_dog', 'car', 'owl',
                'mushroom', 'school_bus', 'whale', 'crocodile', 'roller_coaster', 'octopus', 'basket', 'potato', 'leaf',
                'zigzag', 'syringe', 'pond', 'dragon', 'triangle', 'carrot', 'parachute', 'van', 'stove', 'bridge',
                'swing_set', 'bush', 'The_Great_Wall_of_China', 'helmet', 'flying_saucer', 'trumpet', 'penguin',
                'candle', 'peanut', 'beard', 'fire_hydrant', 'cow', 'lighter', 'waterslide', 'elephant', 'oven',
                'flower', 'spider', 'motorbike', 'strawberry', 'diving_board', 'sweater', 'paper_clip', 'scissors',
                'angel', 'key', 'yoga', 'fork', 'axe', 'rhinoceros', 'brain', 'flip_flops', 'elbow', 'light_bulb',
                'radio', 'camel', 'paintbrush', 'watermelon', 'The_Mona_Lisa', 'stairs', 'picture_frame', 'clock',
                'sun', 'bear', 'bench', 'camera', 'eraser', 'snail', 'beach', 'ocean', 'rifle', 'passport', 'ambulance',
                'lobster', 'flamingo', 'streetlight', 'necklace'],
    'infograph': ['raccoon', 'windmill', 'lightning', 'mouth', 'animal_migration', 'hammer', 'bathtub', 'television',
                  'birthday_cake', 'arm', 'telephone', 'firetruck', 'toothbrush', 'sailboat', 'river', 'baseball',
                  'scorpion', 'golf_club', 'ear', 'traffic_light', 'frying_pan', 'door', 'frog', 'keyboard', 'zebra',
                  'bread', 'umbrella', 'onion', 'stereo', 'dishwasher', 'pants', 'crown', 'hospital', 'square',
                  't-shirt', 'nail', 'squirrel', 'couch', 'hockey_puck', 'laptop', 'dog', 'steak', 'violin', 'cooler',
                  'bracelet', 'crab', 'cell_phone', 'shoe', 'ant', 'asparagus', 'garden_hose', 'rain', 'pineapple',
                  'table', 'rabbit', 'pickup_truck', 'peas', 'ice_cream', 'The_Eiffel_Tower', 'clarinet', 'string_bean',
                  'hedgehog', 'rollerskates', 'ceiling_fan', 'butterfly', 'flashlight', 'crayon', 'palm_tree', 'truck',
                  'shark', 'cat', 'binoculars', 'bee', 'star', 'foot', 'smiley_face', 'envelope', 'teddy-bear',
                  'feather', 'tooth', 'lion', 'toe', 'hand', 'boomerang', 'tree', 'sword', 'diamond', 'pool',
                  'hamburger', 'chandelier', 'floor_lamp', 'circle', 'kangaroo', 'hat', 'snake', 'house', 'sea_turtle',
                  'campfire', 'soccer_ball', 'megaphone', 'grass', 'jacket', 'mountain', 'cookie', 'wine_glass',
                  'octagon', 'church', 'cruise_ship', 'stop_sign', 'knife', 'belt', 'hurricane', 'piano', 'pear',
                  'wheel', 'castle', 'sink', 'headphones', 'bus', 'tennis_racquet', 'shovel', 'moon', 'hot_tub',
                  'trombone', 'pencil', 'duck', 'knee', 'blackberry', 'panda', 'garden', 'chair', 'tractor',
                  'house_plant', 'spreadsheet', 'skateboard', 'bandage', 'tornado', 'anvil', 'squiggle', 'mosquito',
                  'jail', 'snowflake', 'sock', 'bed', 'vase', 'baseball_bat', 'popsicle', 'bowtie', 'calculator',
                  'microphone', 'computer', 'wristwatch', 'shorts', 'dresser', 'mermaid', 'hexagon', 'eyeglasses',
                  'bulldozer', 'harp', 'fish', 'line', 'pliers', 'speedboat', 'toaster', 'banana', 'purse', 'broccoli',
                  'toothpaste', 'dolphin', 'bottlecap', 'washing_machine', 'bird', 'pig', 'pillow', 'parrot', 'giraffe',
                  'lighthouse', 'cello', 'swan', 'donut', 'lantern', 'postcard', 'eye', 'finger', 'coffee_cup',
                  'aircraft_carrier', 'horse', 'paint_can', 'lollipop', 'snowman', 'skyscraper', 'stitches', 'lipstick',
                  'camouflage', 'book', 'rake', 'hot_air_balloon', 'saxophone', 'map', 'matches', 'tiger', 'bat',
                  'cannon', 'hockey_stick', 'power_outlet', 'screwdriver', 'marker', 'see_saw', 'barn', 'bucket',
                  'ladder', 'mouse', 'underwear', 'monkey', 'leg', 'train', 'mailbox', 'basketball', 'moustache',
                  'microwave', 'cactus', 'rainbow', 'fireplace', 'sheep', 'tent', 'fan', 'police_car', 'sandwich',
                  'sleeping_bag', 'guitar', 'mug', 'face', 'compass', 'spoon', 'broom', 'alarm_clock', 'snorkel',
                  'nose', 'hourglass', 'drill', 'cloud', 'cake', 'skull', 'saw', 'goatee', 'cup', 'dumbbell', 'apple',
                  'pizza', 'canoe', 'bicycle', 'grapes', 'toilet', 'blueberry', 'submarine', 'backpack', 'stethoscope',
                  'airplane', 'drums', 'fence', 'teapot', 'remote_control', 'calendar', 'suitcase', 'wine_bottle',
                  'helicopter', 'hot_dog', 'car', 'owl', 'mushroom', 'school_bus', 'whale', 'crocodile',
                  'roller_coaster', 'octopus', 'basket', 'potato', 'leaf', 'zigzag', 'syringe', 'pond', 'dragon',
                  'triangle', 'carrot', 'parachute', 'van', 'stove', 'bridge', 'swing_set', 'bush',
                  'The_Great_Wall_of_China', 'helmet', 'flying_saucer', 'trumpet', 'penguin', 'candle', 'peanut',
                  'beard', 'fire_hydrant', 'cow', 'lighter', 'waterslide', 'elephant', 'oven', 'flower', 'spider',
                  'motorbike', 'strawberry', 'diving_board', 'sweater', 'paper_clip', 'scissors', 'angel', 'key',
                  'yoga', 'fork', 'axe', 'rhinoceros', 'brain', 'flip_flops', 'elbow', 'light_bulb', 'radio', 'camel',
                  'paintbrush', 'watermelon', 'The_Mona_Lisa', 'stairs', 'picture_frame', 'clock', 'sun', 'bear',
                  'bench', 'camera', 'eraser', 'snail', 'beach', 'ocean', 'rifle', 'passport', 'ambulance', 'lobster',
                  'flamingo', 'streetlight', 'necklace'],
    'painting': ['raccoon', 'windmill', 'lightning', 'mouth', 'animal_migration', 'hammer', 'bathtub', 'television',
                 'birthday_cake', 'arm', 'telephone', 'firetruck', 'toothbrush', 'sailboat', 'river', 'baseball',
                 'scorpion', 'golf_club', 'ear', 'traffic_light', 'frying_pan', 'door', 'frog', 'keyboard', 'zebra',
                 'bread', 'umbrella', 'onion', 'stereo', 'dishwasher', 'pants', 'crown', 'hospital', 'square',
                 't-shirt', 'nail', 'squirrel', 'couch', 'hockey_puck', 'laptop', 'dog', 'steak', 'violin', 'cooler',
                 'bracelet', 'crab', 'cell_phone', 'shoe', 'ant', 'asparagus', 'garden_hose', 'rain', 'pineapple',
                 'table', 'rabbit', 'pickup_truck', 'peas', 'ice_cream', 'The_Eiffel_Tower', 'clarinet', 'string_bean',
                 'hedgehog', 'rollerskates', 'ceiling_fan', 'butterfly', 'flashlight', 'crayon', 'palm_tree', 'truck',
                 'shark', 'cat', 'binoculars', 'bee', 'star', 'foot', 'smiley_face', 'envelope', 'teddy-bear',
                 'feather', 'tooth', 'lion', 'toe', 'hand', 'boomerang', 'tree', 'sword', 'diamond', 'pool',
                 'hamburger', 'chandelier', 'floor_lamp', 'circle', 'kangaroo', 'hat', 'snake', 'house', 'sea_turtle',
                 'campfire', 'soccer_ball', 'megaphone', 'grass', 'jacket', 'mountain', 'cookie', 'wine_glass',
                 'octagon', 'church', 'cruise_ship', 'stop_sign', 'knife', 'belt', 'hurricane', 'piano', 'pear',
                 'wheel', 'castle', 'sink', 'headphones', 'bus', 'tennis_racquet', 'shovel', 'moon', 'hot_tub',
                 'trombone', 'pencil', 'duck', 'knee', 'blackberry', 'panda', 'garden', 'chair', 'tractor',
                 'house_plant', 'spreadsheet', 'skateboard', 'bandage', 'tornado', 'anvil', 'squiggle', 'mosquito',
                 'jail', 'snowflake', 'sock', 'bed', 'vase', 'baseball_bat', 'popsicle', 'bowtie', 'calculator',
                 'microphone', 'computer', 'wristwatch', 'shorts', 'dresser', 'mermaid', 'hexagon', 'eyeglasses',
                 'bulldozer', 'harp', 'fish', 'line', 'pliers', 'speedboat', 'toaster', 'banana', 'purse', 'broccoli',
                 'toothpaste', 'dolphin', 'bottlecap', 'washing_machine', 'bird', 'pig', 'pillow', 'parrot', 'giraffe',
                 'lighthouse', 'cello', 'swan', 'donut', 'lantern', 'postcard', 'eye', 'finger', 'coffee_cup',
                 'aircraft_carrier', 'horse', 'paint_can', 'lollipop', 'snowman', 'skyscraper', 'stitches', 'lipstick',
                 'camouflage', 'book', 'rake', 'hot_air_balloon', 'saxophone', 'map', 'matches', 'tiger', 'bat',
                 'cannon', 'hockey_stick', 'power_outlet', 'screwdriver', 'marker', 'see_saw', 'barn', 'bucket',
                 'ladder', 'mouse', 'underwear', 'monkey', 'leg', 'train', 'mailbox', 'basketball', 'moustache',
                 'microwave', 'cactus', 'rainbow', 'fireplace', 'sheep', 'tent', 'fan', 'police_car', 'sandwich',
                 'sleeping_bag', 'guitar', 'mug', 'face', 'compass', 'spoon', 'broom', 'alarm_clock', 'snorkel', 'nose',
                 'hourglass', 'drill', 'cloud', 'cake', 'skull', 'saw', 'goatee', 'cup', 'dumbbell', 'apple', 'pizza',
                 'canoe', 'bicycle', 'grapes', 'toilet', 'blueberry', 'submarine', 'backpack', 'stethoscope',
                 'airplane', 'drums', 'fence', 'teapot', 'remote_control', 'calendar', 'suitcase', 'wine_bottle',
                 'helicopter', 'hot_dog', 'car', 'owl', 'mushroom', 'school_bus', 'whale', 'crocodile',
                 'roller_coaster', 'octopus', 'basket', 'potato', 'leaf', 'zigzag', 'syringe', 'pond', 'dragon',
                 'triangle', 'carrot', 'parachute', 'van', 'stove', 'bridge', 'swing_set', 'bush',
                 'The_Great_Wall_of_China', 'helmet', 'flying_saucer', 'trumpet', 'penguin', 'candle', 'peanut',
                 'beard', 'fire_hydrant', 'cow', 'lighter', 'waterslide', 'elephant', 'oven', 'flower', 'spider',
                 'motorbike', 'strawberry', 'diving_board', 'sweater', 'paper_clip', 'scissors', 'angel', 'key', 'yoga',
                 'fork', 'axe', 'rhinoceros', 'brain', 'flip_flops', 'elbow', 'light_bulb', 'radio', 'camel',
                 'paintbrush', 'watermelon', 'The_Mona_Lisa', 'stairs', 'picture_frame', 'clock', 'sun', 'bear',
                 'bench', 'camera', 'eraser', 'snail', 'beach', 'ocean', 'rifle', 'passport', 'ambulance', 'lobster',
                 'flamingo', 'streetlight', 'necklace'],
    'quickdraw': ['raccoon', 'windmill', 'lightning', 'mouth', 'animal_migration', 'hammer', 'bathtub', 'television',
                  'birthday_cake', 'arm', 'telephone', 'firetruck', 'toothbrush', 'sailboat', 'river', 'baseball',
                  'scorpion', 'golf_club', 'ear', 'traffic_light', 'frying_pan', 'door', 'frog', 'keyboard', 'zebra',
                  'bread', 'umbrella', 'onion', 'stereo', 'dishwasher', 'pants', 'crown', 'hospital', 'square',
                  't-shirt', 'nail', 'squirrel', 'couch', 'hockey_puck', 'laptop', 'dog', 'steak', 'violin', 'cooler',
                  'bracelet', 'crab', 'cell_phone', 'shoe', 'ant', 'asparagus', 'garden_hose', 'rain', 'pineapple',
                  'table', 'rabbit', 'pickup_truck', 'peas', 'ice_cream', 'The_Eiffel_Tower', 'clarinet', 'string_bean',
                  'hedgehog', 'rollerskates', 'ceiling_fan', 'butterfly', 'flashlight', 'crayon', 'palm_tree', 'truck',
                  'shark', 'cat', 'binoculars', 'bee', 'star', 'foot', 'smiley_face', 'envelope', 'teddy-bear',
                  'feather', 'tooth', 'lion', 'toe', 'hand', 'boomerang', 'tree', 'sword', 'diamond', 'pool',
                  'hamburger', 'chandelier', 'floor_lamp', 'circle', 'kangaroo', 'hat', 'snake', 'house', 'sea_turtle',
                  'campfire', 'soccer_ball', 'megaphone', 'grass', 'jacket', 'mountain', 'cookie', 'wine_glass',
                  'octagon', 'church', 'cruise_ship', 'stop_sign', 'knife', 'belt', 'hurricane', 'piano', 'pear',
                  'wheel', 'castle', 'sink', 'headphones', 'bus', 'tennis_racquet', 'shovel', 'moon', 'hot_tub',
                  'trombone', 'pencil', 'duck', 'knee', 'blackberry', 'panda', 'garden', 'chair', 'tractor',
                  'house_plant', 'spreadsheet', 'skateboard', 'bandage', 'tornado', 'anvil', 'squiggle', 'mosquito',
                  'jail', 'snowflake', 'sock', 'bed', 'vase', 'baseball_bat', 'popsicle', 'bowtie', 'calculator',
                  'microphone', 'computer', 'wristwatch', 'shorts', 'dresser', 'mermaid', 'hexagon', 'eyeglasses',
                  'bulldozer', 'harp', 'fish', 'line', 'pliers', 'speedboat', 'toaster', 'banana', 'purse', 'broccoli',
                  'toothpaste', 'dolphin', 'bottlecap', 'washing_machine', 'bird', 'pig', 'pillow', 'parrot', 'giraffe',
                  'lighthouse', 'cello', 'swan', 'donut', 'lantern', 'postcard', 'eye', 'finger', 'coffee_cup',
                  'aircraft_carrier', 'horse', 'paint_can', 'lollipop', 'snowman', 'skyscraper', 'stitches', 'lipstick',
                  'camouflage', 'book', 'rake', 'hot_air_balloon', 'saxophone', 'map', 'matches', 'tiger', 'bat',
                  'cannon', 'hockey_stick', 'power_outlet', 'screwdriver', 'marker', 'see_saw', 'barn', 'bucket',
                  'ladder', 'mouse', 'underwear', 'monkey', 'leg', 'train', 'mailbox', 'basketball', 'moustache',
                  'microwave', 'cactus', 'rainbow', 'fireplace', 'sheep', 'tent', 'fan', 'police_car', 'sandwich',
                  'sleeping_bag', 'guitar', 'mug', 'face', 'compass', 'spoon', 'broom', 'alarm_clock', 'snorkel',
                  'nose', 'hourglass', 'drill', 'cloud', 'cake', 'skull', 'saw', 'goatee', 'cup', 'dumbbell', 'apple',
                  'pizza', 'canoe', 'bicycle', 'grapes', 'toilet', 'blueberry', 'submarine', 'backpack', 'stethoscope',
                  'airplane', 'drums', 'fence', 'teapot', 'remote_control', 'calendar', 'suitcase', 'wine_bottle',
                  'helicopter', 'hot_dog', 'car', 'owl', 'mushroom', 'school_bus', 'whale', 'crocodile',
                  'roller_coaster', 'octopus', 'basket', 'potato', 'leaf', 'zigzag', 'syringe', 'pond', 'dragon',
                  'triangle', 'carrot', 'parachute', 'van', 'stove', 'bridge', 'swing_set', 'bush',
                  'The_Great_Wall_of_China', 'helmet', 'flying_saucer', 'trumpet', 'penguin', 'candle', 'peanut',
                  'beard', 'fire_hydrant', 'cow', 'lighter', 'waterslide', 'elephant', 'oven', 'flower', 'spider',
                  'motorbike', 'strawberry', 'diving_board', 'sweater', 'paper_clip', 'scissors', 'angel', 'key',
                  'yoga', 'fork', 'axe', 'rhinoceros', 'brain', 'flip_flops', 'elbow', 'light_bulb', 'radio', 'camel',
                  'paintbrush', 'watermelon', 'The_Mona_Lisa', 'stairs', 'picture_frame', 'clock', 'sun', 'bear',
                  'bench', 'camera', 'eraser', 'snail', 'beach', 'ocean', 'rifle', 'passport', 'ambulance', 'lobster',
                  'flamingo', 'streetlight', 'necklace'],
    'real': ['raccoon', 'windmill', 'lightning', 'mouth', 'animal_migration', 'hammer', 'bathtub', 'television',
             'birthday_cake', 'arm', 'telephone', 'firetruck', 'toothbrush', 'sailboat', 'river', 'baseball',
             'scorpion', 'golf_club', 'ear', 'traffic_light', 'frying_pan', 'door', 'frog', 'keyboard', 'zebra',
             'bread', 'umbrella', 'onion', 'stereo', 'dishwasher', 'pants', 'crown', 'hospital', 'square', 't-shirt',
             'nail', 'squirrel', 'couch', 'hockey_puck', 'laptop', 'dog', 'steak', 'violin', 'cooler', 'bracelet',
             'crab', 'cell_phone', 'shoe', 'ant', 'asparagus', 'garden_hose', 'rain', 'pineapple', 'table', 'rabbit',
             'pickup_truck', 'peas', 'ice_cream', 'The_Eiffel_Tower', 'clarinet', 'string_bean', 'hedgehog',
             'rollerskates', 'ceiling_fan', 'butterfly', 'flashlight', 'crayon', 'palm_tree', 'truck', 'shark', 'cat',
             'binoculars', 'bee', 'star', 'foot', 'smiley_face', 'envelope', 'teddy-bear', 'feather', 'tooth', 'lion',
             'toe', 'hand', 'boomerang', 'tree', 'sword', 'diamond', 'pool', 'hamburger', 'chandelier', 'floor_lamp',
             'circle', 'kangaroo', 'hat', 'snake', 'house', 'sea_turtle', 'campfire', 'soccer_ball', 'megaphone',
             'grass', 'jacket', 'mountain', 'cookie', 'wine_glass', 'octagon', 'church', 'cruise_ship', 'stop_sign',
             'knife', 'belt', 'hurricane', 'piano', 'pear', 'wheel', 'castle', 'sink', 'headphones', 'bus',
             'tennis_racquet', 'shovel', 'moon', 'hot_tub', 'trombone', 'pencil', 'duck', 'knee', 'blackberry', 'panda',
             'garden', 'chair', 'tractor', 'house_plant', 'spreadsheet', 'skateboard', 'bandage', 'tornado', 'anvil',
             'squiggle', 'mosquito', 'jail', 'snowflake', 'sock', 'bed', 'vase', 'baseball_bat', 'popsicle', 'bowtie',
             'calculator', 'microphone', 'computer', 'wristwatch', 'shorts', 'dresser', 'mermaid', 'hexagon',
             'eyeglasses', 'bulldozer', 'harp', 'fish', 'line', 'pliers', 'speedboat', 'toaster', 'banana', 'purse',
             'broccoli', 'toothpaste', 'dolphin', 'bottlecap', 'washing_machine', 'bird', 'pig', 'pillow', 'parrot',
             'giraffe', 'lighthouse', 'cello', 'swan', 'donut', 'lantern', 'postcard', 'eye', 'finger', 'coffee_cup',
             'aircraft_carrier', 'horse', 'paint_can', 'lollipop', 'snowman', 'skyscraper', 'stitches', 'lipstick',
             'camouflage', 'book', 'rake', 'hot_air_balloon', 'saxophone', 'map', 'matches', 'tiger', 'bat', 'cannon',
             'hockey_stick', 'power_outlet', 'screwdriver', 'marker', 'see_saw', 'barn', 'bucket', 'ladder', 'mouse',
             'underwear', 'monkey', 'leg', 'train', 'mailbox', 'basketball', 'moustache', 'microwave', 'cactus',
             'rainbow', 'fireplace', 'sheep', 'tent', 'fan', 'police_car', 'sandwich', 'sleeping_bag', 'guitar', 'mug',
             'face', 'compass', 'spoon', 'broom', 'alarm_clock', 'snorkel', 'nose', 'hourglass', 'drill', 'cloud',
             'cake', 'skull', 'saw', 'goatee', 'cup', 'dumbbell', 'apple', 'pizza', 'canoe', 'bicycle', 'grapes',
             'toilet', 'blueberry', 'submarine', 'backpack', 'stethoscope', 'airplane', 'drums', 'fence', 'teapot',
             'remote_control', 'calendar', 'suitcase', 'wine_bottle', 'helicopter', 'hot_dog', 'car', 'owl', 'mushroom',
             'school_bus', 'whale', 'crocodile', 'roller_coaster', 'octopus', 'basket', 'potato', 'leaf', 'zigzag',
             'syringe', 'pond', 'dragon', 'triangle', 'carrot', 'parachute', 'van', 'stove', 'bridge', 'swing_set',
             'bush', 'The_Great_Wall_of_China', 'helmet', 'flying_saucer', 'trumpet', 'penguin', 'candle', 'peanut',
             'beard', 'fire_hydrant', 'cow', 'lighter', 'waterslide', 'elephant', 'oven', 'flower', 'spider',
             'motorbike', 'strawberry', 'diving_board', 'sweater', 'paper_clip', 'scissors', 'angel', 'key', 'yoga',
             'fork', 'axe', 'rhinoceros', 'brain', 'flip_flops', 'elbow', 'light_bulb', 'radio', 'camel', 'paintbrush',
             'watermelon', 'The_Mona_Lisa', 'stairs', 'picture_frame', 'clock', 'sun', 'bear', 'bench', 'camera',
             'eraser', 'snail', 'beach', 'ocean', 'rifle', 'passport', 'ambulance', 'lobster', 'flamingo',
             'streetlight', 'necklace']

    }
data_settings['domain-net']['CIQRS_P']['C'] = {
    'clipart': ['raccoon', 'windmill', 'lightning', 'mouth', 'animal_migration', 'hammer', 'bathtub', 'television',
                'birthday_cake', 'arm', 'telephone', 'firetruck', 'toothbrush', 'sailboat', 'river', 'baseball',
                'scorpion', 'golf_club', 'ear', 'traffic_light', 'frying_pan', 'door', 'frog', 'keyboard', 'zebra',
                'bread', 'umbrella', 'onion', 'stereo', 'dishwasher', 'pants', 'crown', 'hospital', 'square', 't-shirt',
                'nail', 'squirrel', 'couch', 'hockey_puck', 'laptop', 'dog', 'steak', 'violin', 'cooler', 'bracelet',
                'crab', 'cell_phone', 'shoe', 'ant', 'asparagus', 'garden_hose', 'rain', 'pineapple', 'table', 'rabbit',
                'pickup_truck', 'peas', 'ice_cream', 'The_Eiffel_Tower', 'clarinet', 'string_bean', 'hedgehog',
                'rollerskates', 'ceiling_fan', 'butterfly', 'flashlight', 'crayon', 'palm_tree', 'truck', 'shark',
                'cat', 'binoculars', 'bee', 'star', 'foot', 'smiley_face', 'envelope', 'teddy-bear', 'feather', 'tooth',
                'lion', 'toe', 'hand', 'boomerang', 'tree', 'sword', 'diamond', 'pool', 'hamburger', 'chandelier',
                'floor_lamp', 'circle', 'kangaroo', 'hat', 'snake', 'house', 'sea_turtle', 'campfire', 'soccer_ball',
                'megaphone', 'grass', 'jacket', 'mountain', 'cookie', 'wine_glass', 'octagon', 'church', 'cruise_ship',
                'stop_sign', 'knife', 'belt', 'hurricane', 'piano', 'pear', 'wheel', 'castle', 'sink', 'headphones',
                'bus', 'tennis_racquet', 'shovel', 'moon', 'hot_tub', 'trombone', 'pencil', 'duck', 'knee',
                'blackberry', 'panda', 'garden', 'chair', 'tractor', 'house_plant', 'spreadsheet', 'skateboard',
                'bandage', 'tornado', 'anvil', 'squiggle', 'mosquito', 'jail', 'snowflake', 'sock', 'bed', 'vase',
                'baseball_bat', 'popsicle', 'bowtie', 'calculator', 'microphone', 'computer', 'wristwatch', 'shorts',
                'dresser', 'mermaid', 'hexagon', 'eyeglasses', 'bulldozer', 'harp', 'fish', 'line', 'pliers',
                'speedboat', 'toaster', 'banana', 'purse', 'broccoli', 'toothpaste', 'dolphin', 'bottlecap',
                'washing_machine', 'bird', 'pig', 'pillow', 'parrot', 'giraffe', 'lighthouse', 'cello', 'swan', 'donut',
                'lantern', 'postcard', 'eye', 'finger', 'coffee_cup', 'aircraft_carrier', 'horse', 'paint_can',
                'lollipop', 'snowman', 'skyscraper', 'stitches', 'lipstick', 'camouflage', 'book', 'rake',
                'hot_air_balloon', 'saxophone', 'map', 'matches', 'tiger', 'bat', 'cannon', 'hockey_stick',
                'power_outlet', 'screwdriver', 'marker', 'see_saw', 'barn', 'bucket', 'ladder', 'mouse', 'underwear',
                'monkey', 'leg', 'train', 'mailbox', 'basketball', 'moustache', 'microwave', 'cactus', 'rainbow',
                'fireplace', 'sheep', 'tent', 'fan', 'police_car', 'sandwich', 'sleeping_bag', 'guitar', 'mug', 'face',
                'compass', 'spoon', 'broom', 'alarm_clock', 'snorkel', 'nose', 'hourglass', 'drill', 'cloud', 'cake',
                'skull', 'saw', 'goatee', 'cup', 'dumbbell', 'apple', 'pizza', 'canoe', 'bicycle', 'grapes', 'toilet',
                'blueberry', 'submarine', 'backpack', 'stethoscope', 'airplane', 'drums', 'fence', 'teapot',
                'remote_control', 'calendar', 'suitcase', 'wine_bottle', 'helicopter', 'hot_dog', 'car', 'owl',
                'mushroom', 'school_bus', 'whale', 'crocodile', 'roller_coaster', 'octopus', 'basket', 'potato', 'leaf',
                'zigzag', 'syringe', 'pond', 'dragon', 'triangle', 'carrot', 'parachute', 'van', 'stove', 'bridge',
                'swing_set', 'bush', 'The_Great_Wall_of_China', 'helmet', 'flying_saucer', 'trumpet', 'penguin',
                'candle', 'peanut', 'beard', 'fire_hydrant', 'cow', 'lighter', 'waterslide', 'elephant', 'oven',
                'flower', 'spider', 'motorbike', 'strawberry', 'diving_board', 'sweater', 'paper_clip', 'scissors',
                'angel', 'key', 'yoga', 'fork', 'axe', 'rhinoceros', 'brain', 'flip_flops', 'elbow', 'light_bulb',
                'radio', 'camel', 'paintbrush', 'watermelon', 'The_Mona_Lisa', 'stairs', 'picture_frame', 'clock',
                'sun', 'bear', 'bench', 'camera', 'eraser', 'snail', 'beach', 'ocean', 'rifle', 'passport', 'ambulance',
                'lobster', 'flamingo', 'streetlight', 'necklace'],
    'infograph': ['raccoon', 'windmill', 'lightning', 'mouth', 'animal_migration', 'hammer', 'bathtub', 'television',
                  'birthday_cake', 'arm', 'telephone', 'firetruck', 'toothbrush', 'sailboat', 'river', 'baseball',
                  'scorpion', 'golf_club', 'ear', 'traffic_light', 'frying_pan', 'door', 'frog', 'keyboard', 'zebra',
                  'bread', 'umbrella', 'onion', 'stereo', 'dishwasher', 'pants', 'crown', 'hospital', 'square',
                  't-shirt', 'nail', 'squirrel', 'couch', 'hockey_puck', 'laptop', 'dog', 'steak', 'violin', 'cooler',
                  'bracelet', 'crab', 'cell_phone', 'shoe', 'ant', 'asparagus', 'garden_hose', 'rain', 'pineapple',
                  'table', 'rabbit', 'pickup_truck', 'peas', 'ice_cream', 'The_Eiffel_Tower', 'clarinet', 'string_bean',
                  'hedgehog', 'rollerskates', 'ceiling_fan', 'butterfly', 'flashlight', 'crayon', 'palm_tree', 'truck',
                  'shark', 'cat', 'binoculars', 'bee', 'star', 'foot', 'smiley_face', 'envelope', 'teddy-bear',
                  'feather', 'tooth', 'lion', 'toe', 'hand', 'boomerang', 'tree', 'sword', 'diamond', 'pool',
                  'hamburger', 'chandelier', 'floor_lamp', 'circle', 'kangaroo', 'hat', 'snake', 'house', 'sea_turtle',
                  'campfire', 'soccer_ball', 'megaphone', 'grass', 'jacket', 'mountain', 'cookie', 'wine_glass',
                  'octagon', 'church', 'cruise_ship', 'stop_sign', 'knife', 'belt', 'hurricane', 'piano', 'pear',
                  'wheel', 'castle', 'sink', 'headphones', 'bus', 'tennis_racquet', 'shovel', 'moon', 'hot_tub',
                  'trombone', 'pencil', 'duck', 'knee', 'blackberry', 'panda', 'garden', 'chair', 'tractor',
                  'house_plant', 'spreadsheet', 'skateboard', 'bandage', 'tornado', 'anvil', 'squiggle', 'mosquito',
                  'jail', 'snowflake', 'sock', 'bed', 'vase', 'baseball_bat', 'popsicle', 'bowtie', 'calculator',
                  'microphone', 'computer', 'wristwatch', 'shorts', 'dresser', 'mermaid', 'hexagon', 'eyeglasses',
                  'bulldozer', 'harp', 'fish', 'line', 'pliers', 'speedboat', 'toaster', 'banana', 'purse', 'broccoli',
                  'toothpaste', 'dolphin', 'bottlecap', 'washing_machine', 'bird', 'pig', 'pillow', 'parrot', 'giraffe',
                  'lighthouse', 'cello', 'swan', 'donut', 'lantern', 'postcard', 'eye', 'finger', 'coffee_cup',
                  'aircraft_carrier', 'horse', 'paint_can', 'lollipop', 'snowman', 'skyscraper', 'stitches', 'lipstick',
                  'camouflage', 'book', 'rake', 'hot_air_balloon', 'saxophone', 'map', 'matches', 'tiger', 'bat',
                  'cannon', 'hockey_stick', 'power_outlet', 'screwdriver', 'marker', 'see_saw', 'barn', 'bucket',
                  'ladder', 'mouse', 'underwear', 'monkey', 'leg', 'train', 'mailbox', 'basketball', 'moustache',
                  'microwave', 'cactus', 'rainbow', 'fireplace', 'sheep', 'tent', 'fan', 'police_car', 'sandwich',
                  'sleeping_bag', 'guitar', 'mug', 'face', 'compass', 'spoon', 'broom', 'alarm_clock', 'snorkel',
                  'nose', 'hourglass', 'drill', 'cloud', 'cake', 'skull', 'saw', 'goatee', 'cup', 'dumbbell', 'apple',
                  'pizza', 'canoe', 'bicycle', 'grapes', 'toilet', 'blueberry', 'submarine', 'backpack', 'stethoscope',
                  'airplane', 'drums', 'fence', 'teapot', 'remote_control', 'calendar', 'suitcase', 'wine_bottle',
                  'helicopter', 'hot_dog', 'car', 'owl', 'mushroom', 'school_bus', 'whale', 'crocodile',
                  'roller_coaster', 'octopus', 'basket', 'potato', 'leaf', 'zigzag', 'syringe', 'pond', 'dragon',
                  'triangle', 'carrot', 'parachute', 'van', 'stove', 'bridge', 'swing_set', 'bush',
                  'The_Great_Wall_of_China', 'helmet', 'flying_saucer', 'trumpet', 'penguin', 'candle', 'peanut',
                  'beard', 'fire_hydrant', 'cow', 'lighter', 'waterslide', 'elephant', 'oven', 'flower', 'spider',
                  'motorbike', 'strawberry', 'diving_board', 'sweater', 'paper_clip', 'scissors', 'angel', 'key',
                  'yoga', 'fork', 'axe', 'rhinoceros', 'brain', 'flip_flops', 'elbow', 'light_bulb', 'radio', 'camel',
                  'paintbrush', 'watermelon', 'The_Mona_Lisa', 'stairs', 'picture_frame', 'clock', 'sun', 'bear',
                  'bench', 'camera', 'eraser', 'snail', 'beach', 'ocean', 'rifle', 'passport', 'ambulance', 'lobster',
                  'flamingo', 'streetlight', 'necklace'],
    'quickdraw': ['raccoon', 'windmill', 'lightning', 'mouth', 'animal_migration', 'hammer', 'bathtub', 'television',
                  'birthday_cake', 'arm', 'telephone', 'firetruck', 'toothbrush', 'sailboat', 'river', 'baseball',
                  'scorpion', 'golf_club', 'ear', 'traffic_light', 'frying_pan', 'door', 'frog', 'keyboard', 'zebra',
                  'bread', 'umbrella', 'onion', 'stereo', 'dishwasher', 'pants', 'crown', 'hospital', 'square',
                  't-shirt', 'nail', 'squirrel', 'couch', 'hockey_puck', 'laptop', 'dog', 'steak', 'violin', 'cooler',
                  'bracelet', 'crab', 'cell_phone', 'shoe', 'ant', 'asparagus', 'garden_hose', 'rain', 'pineapple',
                  'table', 'rabbit', 'pickup_truck', 'peas', 'ice_cream', 'The_Eiffel_Tower', 'clarinet', 'string_bean',
                  'hedgehog', 'rollerskates', 'ceiling_fan', 'butterfly', 'flashlight', 'crayon', 'palm_tree', 'truck',
                  'shark', 'cat', 'binoculars', 'bee', 'star', 'foot', 'smiley_face', 'envelope', 'teddy-bear',
                  'feather', 'tooth', 'lion', 'toe', 'hand', 'boomerang', 'tree', 'sword', 'diamond', 'pool',
                  'hamburger', 'chandelier', 'floor_lamp', 'circle', 'kangaroo', 'hat', 'snake', 'house', 'sea_turtle',
                  'campfire', 'soccer_ball', 'megaphone', 'grass', 'jacket', 'mountain', 'cookie', 'wine_glass',
                  'octagon', 'church', 'cruise_ship', 'stop_sign', 'knife', 'belt', 'hurricane', 'piano', 'pear',
                  'wheel', 'castle', 'sink', 'headphones', 'bus', 'tennis_racquet', 'shovel', 'moon', 'hot_tub',
                  'trombone', 'pencil', 'duck', 'knee', 'blackberry', 'panda', 'garden', 'chair', 'tractor',
                  'house_plant', 'spreadsheet', 'skateboard', 'bandage', 'tornado', 'anvil', 'squiggle', 'mosquito',
                  'jail', 'snowflake', 'sock', 'bed', 'vase', 'baseball_bat', 'popsicle', 'bowtie', 'calculator',
                  'microphone', 'computer', 'wristwatch', 'shorts', 'dresser', 'mermaid', 'hexagon', 'eyeglasses',
                  'bulldozer', 'harp', 'fish', 'line', 'pliers', 'speedboat', 'toaster', 'banana', 'purse', 'broccoli',
                  'toothpaste', 'dolphin', 'bottlecap', 'washing_machine', 'bird', 'pig', 'pillow', 'parrot', 'giraffe',
                  'lighthouse', 'cello', 'swan', 'donut', 'lantern', 'postcard', 'eye', 'finger', 'coffee_cup',
                  'aircraft_carrier', 'horse', 'paint_can', 'lollipop', 'snowman', 'skyscraper', 'stitches', 'lipstick',
                  'camouflage', 'book', 'rake', 'hot_air_balloon', 'saxophone', 'map', 'matches', 'tiger', 'bat',
                  'cannon', 'hockey_stick', 'power_outlet', 'screwdriver', 'marker', 'see_saw', 'barn', 'bucket',
                  'ladder', 'mouse', 'underwear', 'monkey', 'leg', 'train', 'mailbox', 'basketball', 'moustache',
                  'microwave', 'cactus', 'rainbow', 'fireplace', 'sheep', 'tent', 'fan', 'police_car', 'sandwich',
                  'sleeping_bag', 'guitar', 'mug', 'face', 'compass', 'spoon', 'broom', 'alarm_clock', 'snorkel',
                  'nose', 'hourglass', 'drill', 'cloud', 'cake', 'skull', 'saw', 'goatee', 'cup', 'dumbbell', 'apple',
                  'pizza', 'canoe', 'bicycle', 'grapes', 'toilet', 'blueberry', 'submarine', 'backpack', 'stethoscope',
                  'airplane', 'drums', 'fence', 'teapot', 'remote_control', 'calendar', 'suitcase', 'wine_bottle',
                  'helicopter', 'hot_dog', 'car', 'owl', 'mushroom', 'school_bus', 'whale', 'crocodile',
                  'roller_coaster', 'octopus', 'basket', 'potato', 'leaf', 'zigzag', 'syringe', 'pond', 'dragon',
                  'triangle', 'carrot', 'parachute', 'van', 'stove', 'bridge', 'swing_set', 'bush',
                  'The_Great_Wall_of_China', 'helmet', 'flying_saucer', 'trumpet', 'penguin', 'candle', 'peanut',
                  'beard', 'fire_hydrant', 'cow', 'lighter', 'waterslide', 'elephant', 'oven', 'flower', 'spider',
                  'motorbike', 'strawberry', 'diving_board', 'sweater', 'paper_clip', 'scissors', 'angel', 'key',
                  'yoga', 'fork', 'axe', 'rhinoceros', 'brain', 'flip_flops', 'elbow', 'light_bulb', 'radio', 'camel',
                  'paintbrush', 'watermelon', 'The_Mona_Lisa', 'stairs', 'picture_frame', 'clock', 'sun', 'bear',
                  'bench', 'camera', 'eraser', 'snail', 'beach', 'ocean', 'rifle', 'passport', 'ambulance', 'lobster',
                  'flamingo', 'streetlight', 'necklace'],
    'real': ['raccoon', 'windmill', 'lightning', 'mouth', 'animal_migration', 'hammer', 'bathtub', 'television',
             'birthday_cake', 'arm', 'telephone', 'firetruck', 'toothbrush', 'sailboat', 'river', 'baseball',
             'scorpion', 'golf_club', 'ear', 'traffic_light', 'frying_pan', 'door', 'frog', 'keyboard', 'zebra',
             'bread', 'umbrella', 'onion', 'stereo', 'dishwasher', 'pants', 'crown', 'hospital', 'square', 't-shirt',
             'nail', 'squirrel', 'couch', 'hockey_puck', 'laptop', 'dog', 'steak', 'violin', 'cooler', 'bracelet',
             'crab', 'cell_phone', 'shoe', 'ant', 'asparagus', 'garden_hose', 'rain', 'pineapple', 'table', 'rabbit',
             'pickup_truck', 'peas', 'ice_cream', 'The_Eiffel_Tower', 'clarinet', 'string_bean', 'hedgehog',
             'rollerskates', 'ceiling_fan', 'butterfly', 'flashlight', 'crayon', 'palm_tree', 'truck', 'shark', 'cat',
             'binoculars', 'bee', 'star', 'foot', 'smiley_face', 'envelope', 'teddy-bear', 'feather', 'tooth', 'lion',
             'toe', 'hand', 'boomerang', 'tree', 'sword', 'diamond', 'pool', 'hamburger', 'chandelier', 'floor_lamp',
             'circle', 'kangaroo', 'hat', 'snake', 'house', 'sea_turtle', 'campfire', 'soccer_ball', 'megaphone',
             'grass', 'jacket', 'mountain', 'cookie', 'wine_glass', 'octagon', 'church', 'cruise_ship', 'stop_sign',
             'knife', 'belt', 'hurricane', 'piano', 'pear', 'wheel', 'castle', 'sink', 'headphones', 'bus',
             'tennis_racquet', 'shovel', 'moon', 'hot_tub', 'trombone', 'pencil', 'duck', 'knee', 'blackberry', 'panda',
             'garden', 'chair', 'tractor', 'house_plant', 'spreadsheet', 'skateboard', 'bandage', 'tornado', 'anvil',
             'squiggle', 'mosquito', 'jail', 'snowflake', 'sock', 'bed', 'vase', 'baseball_bat', 'popsicle', 'bowtie',
             'calculator', 'microphone', 'computer', 'wristwatch', 'shorts', 'dresser', 'mermaid', 'hexagon',
             'eyeglasses', 'bulldozer', 'harp', 'fish', 'line', 'pliers', 'speedboat', 'toaster', 'banana', 'purse',
             'broccoli', 'toothpaste', 'dolphin', 'bottlecap', 'washing_machine', 'bird', 'pig', 'pillow', 'parrot',
             'giraffe', 'lighthouse', 'cello', 'swan', 'donut', 'lantern', 'postcard', 'eye', 'finger', 'coffee_cup',
             'aircraft_carrier', 'horse', 'paint_can', 'lollipop', 'snowman', 'skyscraper', 'stitches', 'lipstick',
             'camouflage', 'book', 'rake', 'hot_air_balloon', 'saxophone', 'map', 'matches', 'tiger', 'bat', 'cannon',
             'hockey_stick', 'power_outlet', 'screwdriver', 'marker', 'see_saw', 'barn', 'bucket', 'ladder', 'mouse',
             'underwear', 'monkey', 'leg', 'train', 'mailbox', 'basketball', 'moustache', 'microwave', 'cactus',
             'rainbow', 'fireplace', 'sheep', 'tent', 'fan', 'police_car', 'sandwich', 'sleeping_bag', 'guitar', 'mug',
             'face', 'compass', 'spoon', 'broom', 'alarm_clock', 'snorkel', 'nose', 'hourglass', 'drill', 'cloud',
             'cake', 'skull', 'saw', 'goatee', 'cup', 'dumbbell', 'apple', 'pizza', 'canoe', 'bicycle', 'grapes',
             'toilet', 'blueberry', 'submarine', 'backpack', 'stethoscope', 'airplane', 'drums', 'fence', 'teapot',
             'remote_control', 'calendar', 'suitcase', 'wine_bottle', 'helicopter', 'hot_dog', 'car', 'owl', 'mushroom',
             'school_bus', 'whale', 'crocodile', 'roller_coaster', 'octopus', 'basket', 'potato', 'leaf', 'zigzag',
             'syringe', 'pond', 'dragon', 'triangle', 'carrot', 'parachute', 'van', 'stove', 'bridge', 'swing_set',
             'bush', 'The_Great_Wall_of_China', 'helmet', 'flying_saucer', 'trumpet', 'penguin', 'candle', 'peanut',
             'beard', 'fire_hydrant', 'cow', 'lighter', 'waterslide', 'elephant', 'oven', 'flower', 'spider',
             'motorbike', 'strawberry', 'diving_board', 'sweater', 'paper_clip', 'scissors', 'angel', 'key', 'yoga',
             'fork', 'axe', 'rhinoceros', 'brain', 'flip_flops', 'elbow', 'light_bulb', 'radio', 'camel', 'paintbrush',
             'watermelon', 'The_Mona_Lisa', 'stairs', 'picture_frame', 'clock', 'sun', 'bear', 'bench', 'camera',
             'eraser', 'snail', 'beach', 'ocean', 'rifle', 'passport', 'ambulance', 'lobster', 'flamingo',
             'streetlight', 'necklace'],
    'sketch': ['raccoon', 'windmill', 'lightning', 'mouth', 'animal_migration', 'hammer', 'bathtub', 'television',
               'birthday_cake', 'arm', 'telephone', 'firetruck', 'toothbrush', 'sailboat', 'river', 'baseball',
               'scorpion', 'golf_club', 'ear', 'traffic_light', 'frying_pan', 'door', 'frog', 'keyboard', 'zebra',
               'bread', 'umbrella', 'onion', 'stereo', 'dishwasher', 'pants', 'crown', 'hospital', 'square', 't-shirt',
               'nail', 'squirrel', 'couch', 'hockey_puck', 'laptop', 'dog', 'steak', 'violin', 'cooler', 'bracelet',
               'crab', 'cell_phone', 'shoe', 'ant', 'asparagus', 'garden_hose', 'rain', 'pineapple', 'table', 'rabbit',
               'pickup_truck', 'peas', 'ice_cream', 'The_Eiffel_Tower', 'clarinet', 'string_bean', 'hedgehog',
               'rollerskates', 'ceiling_fan', 'butterfly', 'flashlight', 'crayon', 'palm_tree', 'truck', 'shark', 'cat',
               'binoculars', 'bee', 'star', 'foot', 'smiley_face', 'envelope', 'teddy-bear', 'feather', 'tooth', 'lion',
               'toe', 'hand', 'boomerang', 'tree', 'sword', 'diamond', 'pool', 'hamburger', 'chandelier', 'floor_lamp',
               'circle', 'kangaroo', 'hat', 'snake', 'house', 'sea_turtle', 'campfire', 'soccer_ball', 'megaphone',
               'grass', 'jacket', 'mountain', 'cookie', 'wine_glass', 'octagon', 'church', 'cruise_ship', 'stop_sign',
               'knife', 'belt', 'hurricane', 'piano', 'pear', 'wheel', 'castle', 'sink', 'headphones', 'bus',
               'tennis_racquet', 'shovel', 'moon', 'hot_tub', 'trombone', 'pencil', 'duck', 'knee', 'blackberry',
               'panda', 'garden', 'chair', 'tractor', 'house_plant', 'spreadsheet', 'skateboard', 'bandage', 'tornado',
               'anvil', 'squiggle', 'mosquito', 'jail', 'snowflake', 'sock', 'bed', 'vase', 'baseball_bat', 'popsicle',
               'bowtie', 'calculator', 'microphone', 'computer', 'wristwatch', 'shorts', 'dresser', 'mermaid',
               'hexagon', 'eyeglasses', 'bulldozer', 'harp', 'fish', 'line', 'pliers', 'speedboat', 'toaster', 'banana',
               'purse', 'broccoli', 'toothpaste', 'dolphin', 'bottlecap', 'washing_machine', 'bird', 'pig', 'pillow',
               'parrot', 'giraffe', 'lighthouse', 'cello', 'swan', 'donut', 'lantern', 'postcard', 'eye', 'finger',
               'coffee_cup', 'aircraft_carrier', 'horse', 'paint_can', 'lollipop', 'snowman', 'skyscraper', 'stitches',
               'lipstick', 'camouflage', 'book', 'rake', 'hot_air_balloon', 'saxophone', 'map', 'matches', 'tiger',
               'bat', 'cannon', 'hockey_stick', 'power_outlet', 'screwdriver', 'marker', 'see_saw', 'barn', 'bucket',
               'ladder', 'mouse', 'underwear', 'monkey', 'leg', 'train', 'mailbox', 'basketball', 'moustache',
               'microwave', 'cactus', 'rainbow', 'fireplace', 'sheep', 'tent', 'fan', 'police_car', 'sandwich',
               'sleeping_bag', 'guitar', 'mug', 'face', 'compass', 'spoon', 'broom', 'alarm_clock', 'snorkel', 'nose',
               'hourglass', 'drill', 'cloud', 'cake', 'skull', 'saw', 'goatee', 'cup', 'dumbbell', 'apple', 'pizza',
               'canoe', 'bicycle', 'grapes', 'toilet', 'blueberry', 'submarine', 'backpack', 'stethoscope', 'airplane',
               'drums', 'fence', 'teapot', 'remote_control', 'calendar', 'suitcase', 'wine_bottle', 'helicopter',
               'hot_dog', 'car', 'owl', 'mushroom', 'school_bus', 'whale', 'crocodile', 'roller_coaster', 'octopus',
               'basket', 'potato', 'leaf', 'zigzag', 'syringe', 'pond', 'dragon', 'triangle', 'carrot', 'parachute',
               'van', 'stove', 'bridge', 'swing_set', 'bush', 'The_Great_Wall_of_China', 'helmet', 'flying_saucer',
               'trumpet', 'penguin', 'candle', 'peanut', 'beard', 'fire_hydrant', 'cow', 'lighter', 'waterslide',
               'elephant', 'oven', 'flower', 'spider', 'motorbike', 'strawberry', 'diving_board', 'sweater',
               'paper_clip', 'scissors', 'angel', 'key', 'yoga', 'fork', 'axe', 'rhinoceros', 'brain', 'flip_flops',
               'elbow', 'light_bulb', 'radio', 'camel', 'paintbrush', 'watermelon', 'The_Mona_Lisa', 'stairs',
               'picture_frame', 'clock', 'sun', 'bear', 'bench', 'camera', 'eraser', 'snail', 'beach', 'ocean', 'rifle',
               'passport', 'ambulance', 'lobster', 'flamingo', 'streetlight', 'necklace']
    }
data_settings['domain-net']['IPQRS_C']['C'] = {
    'infograph': ['raccoon', 'windmill', 'lightning', 'mouth', 'animal_migration', 'hammer', 'bathtub', 'television',
                  'birthday_cake', 'arm', 'telephone', 'firetruck', 'toothbrush', 'sailboat', 'river', 'baseball',
                  'scorpion', 'golf_club', 'ear', 'traffic_light', 'frying_pan', 'door', 'frog', 'keyboard', 'zebra',
                  'bread', 'umbrella', 'onion', 'stereo', 'dishwasher', 'pants', 'crown', 'hospital', 'square',
                  't-shirt', 'nail', 'squirrel', 'couch', 'hockey_puck', 'laptop', 'dog', 'steak', 'violin', 'cooler',
                  'bracelet', 'crab', 'cell_phone', 'shoe', 'ant', 'asparagus', 'garden_hose', 'rain', 'pineapple',
                  'table', 'rabbit', 'pickup_truck', 'peas', 'ice_cream', 'The_Eiffel_Tower', 'clarinet', 'string_bean',
                  'hedgehog', 'rollerskates', 'ceiling_fan', 'butterfly', 'flashlight', 'crayon', 'palm_tree', 'truck',
                  'shark', 'cat', 'binoculars', 'bee', 'star', 'foot', 'smiley_face', 'envelope', 'teddy-bear',
                  'feather', 'tooth', 'lion', 'toe', 'hand', 'boomerang', 'tree', 'sword', 'diamond', 'pool',
                  'hamburger', 'chandelier', 'floor_lamp', 'circle', 'kangaroo', 'hat', 'snake', 'house', 'sea_turtle',
                  'campfire', 'soccer_ball', 'megaphone', 'grass', 'jacket', 'mountain', 'cookie', 'wine_glass',
                  'octagon', 'church', 'cruise_ship', 'stop_sign', 'knife', 'belt', 'hurricane', 'piano', 'pear',
                  'wheel', 'castle', 'sink', 'headphones', 'bus', 'tennis_racquet', 'shovel', 'moon', 'hot_tub',
                  'trombone', 'pencil', 'duck', 'knee', 'blackberry', 'panda', 'garden', 'chair', 'tractor',
                  'house_plant', 'spreadsheet', 'skateboard', 'bandage', 'tornado', 'anvil', 'squiggle', 'mosquito',
                  'jail', 'snowflake', 'sock', 'bed', 'vase', 'baseball_bat', 'popsicle', 'bowtie', 'calculator',
                  'microphone', 'computer', 'wristwatch', 'shorts', 'dresser', 'mermaid', 'hexagon', 'eyeglasses',
                  'bulldozer', 'harp', 'fish', 'line', 'pliers', 'speedboat', 'toaster', 'banana', 'purse', 'broccoli',
                  'toothpaste', 'dolphin', 'bottlecap', 'washing_machine', 'bird', 'pig', 'pillow', 'parrot', 'giraffe',
                  'lighthouse', 'cello', 'swan', 'donut', 'lantern', 'postcard', 'eye', 'finger', 'coffee_cup',
                  'aircraft_carrier', 'horse', 'paint_can', 'lollipop', 'snowman', 'skyscraper', 'stitches', 'lipstick',
                  'camouflage', 'book', 'rake', 'hot_air_balloon', 'saxophone', 'map', 'matches', 'tiger', 'bat',
                  'cannon', 'hockey_stick', 'power_outlet', 'screwdriver', 'marker', 'see_saw', 'barn', 'bucket',
                  'ladder', 'mouse', 'underwear', 'monkey', 'leg', 'train', 'mailbox', 'basketball', 'moustache',
                  'microwave', 'cactus', 'rainbow', 'fireplace', 'sheep', 'tent', 'fan', 'police_car', 'sandwich',
                  'sleeping_bag', 'guitar', 'mug', 'face', 'compass', 'spoon', 'broom', 'alarm_clock', 'snorkel',
                  'nose', 'hourglass', 'drill', 'cloud', 'cake', 'skull', 'saw', 'goatee', 'cup', 'dumbbell', 'apple',
                  'pizza', 'canoe', 'bicycle', 'grapes', 'toilet', 'blueberry', 'submarine', 'backpack', 'stethoscope',
                  'airplane', 'drums', 'fence', 'teapot', 'remote_control', 'calendar', 'suitcase', 'wine_bottle',
                  'helicopter', 'hot_dog', 'car', 'owl', 'mushroom', 'school_bus', 'whale', 'crocodile',
                  'roller_coaster', 'octopus', 'basket', 'potato', 'leaf', 'zigzag', 'syringe', 'pond', 'dragon',
                  'triangle', 'carrot', 'parachute', 'van', 'stove', 'bridge', 'swing_set', 'bush',
                  'The_Great_Wall_of_China', 'helmet', 'flying_saucer', 'trumpet', 'penguin', 'candle', 'peanut',
                  'beard', 'fire_hydrant', 'cow', 'lighter', 'waterslide', 'elephant', 'oven', 'flower', 'spider',
                  'motorbike', 'strawberry', 'diving_board', 'sweater', 'paper_clip', 'scissors', 'angel', 'key',
                  'yoga', 'fork', 'axe', 'rhinoceros', 'brain', 'flip_flops', 'elbow', 'light_bulb', 'radio', 'camel',
                  'paintbrush', 'watermelon', 'The_Mona_Lisa', 'stairs', 'picture_frame', 'clock', 'sun', 'bear',
                  'bench', 'camera', 'eraser', 'snail', 'beach', 'ocean', 'rifle', 'passport', 'ambulance', 'lobster',
                  'flamingo', 'streetlight', 'necklace'],
    'painting': ['raccoon', 'windmill', 'lightning', 'mouth', 'animal_migration', 'hammer', 'bathtub', 'television',
                 'birthday_cake', 'arm', 'telephone', 'firetruck', 'toothbrush', 'sailboat', 'river', 'baseball',
                 'scorpion', 'golf_club', 'ear', 'traffic_light', 'frying_pan', 'door', 'frog', 'keyboard', 'zebra',
                 'bread', 'umbrella', 'onion', 'stereo', 'dishwasher', 'pants', 'crown', 'hospital', 'square',
                 't-shirt', 'nail', 'squirrel', 'couch', 'hockey_puck', 'laptop', 'dog', 'steak', 'violin', 'cooler',
                 'bracelet', 'crab', 'cell_phone', 'shoe', 'ant', 'asparagus', 'garden_hose', 'rain', 'pineapple',
                 'table', 'rabbit', 'pickup_truck', 'peas', 'ice_cream', 'The_Eiffel_Tower', 'clarinet', 'string_bean',
                 'hedgehog', 'rollerskates', 'ceiling_fan', 'butterfly', 'flashlight', 'crayon', 'palm_tree', 'truck',
                 'shark', 'cat', 'binoculars', 'bee', 'star', 'foot', 'smiley_face', 'envelope', 'teddy-bear',
                 'feather', 'tooth', 'lion', 'toe', 'hand', 'boomerang', 'tree', 'sword', 'diamond', 'pool',
                 'hamburger', 'chandelier', 'floor_lamp', 'circle', 'kangaroo', 'hat', 'snake', 'house', 'sea_turtle',
                 'campfire', 'soccer_ball', 'megaphone', 'grass', 'jacket', 'mountain', 'cookie', 'wine_glass',
                 'octagon', 'church', 'cruise_ship', 'stop_sign', 'knife', 'belt', 'hurricane', 'piano', 'pear',
                 'wheel', 'castle', 'sink', 'headphones', 'bus', 'tennis_racquet', 'shovel', 'moon', 'hot_tub',
                 'trombone', 'pencil', 'duck', 'knee', 'blackberry', 'panda', 'garden', 'chair', 'tractor',
                 'house_plant', 'spreadsheet', 'skateboard', 'bandage', 'tornado', 'anvil', 'squiggle', 'mosquito',
                 'jail', 'snowflake', 'sock', 'bed', 'vase', 'baseball_bat', 'popsicle', 'bowtie', 'calculator',
                 'microphone', 'computer', 'wristwatch', 'shorts', 'dresser', 'mermaid', 'hexagon', 'eyeglasses',
                 'bulldozer', 'harp', 'fish', 'line', 'pliers', 'speedboat', 'toaster', 'banana', 'purse', 'broccoli',
                 'toothpaste', 'dolphin', 'bottlecap', 'washing_machine', 'bird', 'pig', 'pillow', 'parrot', 'giraffe',
                 'lighthouse', 'cello', 'swan', 'donut', 'lantern', 'postcard', 'eye', 'finger', 'coffee_cup',
                 'aircraft_carrier', 'horse', 'paint_can', 'lollipop', 'snowman', 'skyscraper', 'stitches', 'lipstick',
                 'camouflage', 'book', 'rake', 'hot_air_balloon', 'saxophone', 'map', 'matches', 'tiger', 'bat',
                 'cannon', 'hockey_stick', 'power_outlet', 'screwdriver', 'marker', 'see_saw', 'barn', 'bucket',
                 'ladder', 'mouse', 'underwear', 'monkey', 'leg', 'train', 'mailbox', 'basketball', 'moustache',
                 'microwave', 'cactus', 'rainbow', 'fireplace', 'sheep', 'tent', 'fan', 'police_car', 'sandwich',
                 'sleeping_bag', 'guitar', 'mug', 'face', 'compass', 'spoon', 'broom', 'alarm_clock', 'snorkel', 'nose',
                 'hourglass', 'drill', 'cloud', 'cake', 'skull', 'saw', 'goatee', 'cup', 'dumbbell', 'apple', 'pizza',
                 'canoe', 'bicycle', 'grapes', 'toilet', 'blueberry', 'submarine', 'backpack', 'stethoscope',
                 'airplane', 'drums', 'fence', 'teapot', 'remote_control', 'calendar', 'suitcase', 'wine_bottle',
                 'helicopter', 'hot_dog', 'car', 'owl', 'mushroom', 'school_bus', 'whale', 'crocodile',
                 'roller_coaster', 'octopus', 'basket', 'potato', 'leaf', 'zigzag', 'syringe', 'pond', 'dragon',
                 'triangle', 'carrot', 'parachute', 'van', 'stove', 'bridge', 'swing_set', 'bush',
                 'The_Great_Wall_of_China', 'helmet', 'flying_saucer', 'trumpet', 'penguin', 'candle', 'peanut',
                 'beard', 'fire_hydrant', 'cow', 'lighter', 'waterslide', 'elephant', 'oven', 'flower', 'spider',
                 'motorbike', 'strawberry', 'diving_board', 'sweater', 'paper_clip', 'scissors', 'angel', 'key', 'yoga',
                 'fork', 'axe', 'rhinoceros', 'brain', 'flip_flops', 'elbow', 'light_bulb', 'radio', 'camel',
                 'paintbrush', 'watermelon', 'The_Mona_Lisa', 'stairs', 'picture_frame', 'clock', 'sun', 'bear',
                 'bench', 'camera', 'eraser', 'snail', 'beach', 'ocean', 'rifle', 'passport', 'ambulance', 'lobster',
                 'flamingo', 'streetlight', 'necklace'],
    'quickdraw': ['raccoon', 'windmill', 'lightning', 'mouth', 'animal_migration', 'hammer', 'bathtub', 'television',
                  'birthday_cake', 'arm', 'telephone', 'firetruck', 'toothbrush', 'sailboat', 'river', 'baseball',
                  'scorpion', 'golf_club', 'ear', 'traffic_light', 'frying_pan', 'door', 'frog', 'keyboard', 'zebra',
                  'bread', 'umbrella', 'onion', 'stereo', 'dishwasher', 'pants', 'crown', 'hospital', 'square',
                  't-shirt', 'nail', 'squirrel', 'couch', 'hockey_puck', 'laptop', 'dog', 'steak', 'violin', 'cooler',
                  'bracelet', 'crab', 'cell_phone', 'shoe', 'ant', 'asparagus', 'garden_hose', 'rain', 'pineapple',
                  'table', 'rabbit', 'pickup_truck', 'peas', 'ice_cream', 'The_Eiffel_Tower', 'clarinet', 'string_bean',
                  'hedgehog', 'rollerskates', 'ceiling_fan', 'butterfly', 'flashlight', 'crayon', 'palm_tree', 'truck',
                  'shark', 'cat', 'binoculars', 'bee', 'star', 'foot', 'smiley_face', 'envelope', 'teddy-bear',
                  'feather', 'tooth', 'lion', 'toe', 'hand', 'boomerang', 'tree', 'sword', 'diamond', 'pool',
                  'hamburger', 'chandelier', 'floor_lamp', 'circle', 'kangaroo', 'hat', 'snake', 'house', 'sea_turtle',
                  'campfire', 'soccer_ball', 'megaphone', 'grass', 'jacket', 'mountain', 'cookie', 'wine_glass',
                  'octagon', 'church', 'cruise_ship', 'stop_sign', 'knife', 'belt', 'hurricane', 'piano', 'pear',
                  'wheel', 'castle', 'sink', 'headphones', 'bus', 'tennis_racquet', 'shovel', 'moon', 'hot_tub',
                  'trombone', 'pencil', 'duck', 'knee', 'blackberry', 'panda', 'garden', 'chair', 'tractor',
                  'house_plant', 'spreadsheet', 'skateboard', 'bandage', 'tornado', 'anvil', 'squiggle', 'mosquito',
                  'jail', 'snowflake', 'sock', 'bed', 'vase', 'baseball_bat', 'popsicle', 'bowtie', 'calculator',
                  'microphone', 'computer', 'wristwatch', 'shorts', 'dresser', 'mermaid', 'hexagon', 'eyeglasses',
                  'bulldozer', 'harp', 'fish', 'line', 'pliers', 'speedboat', 'toaster', 'banana', 'purse', 'broccoli',
                  'toothpaste', 'dolphin', 'bottlecap', 'washing_machine', 'bird', 'pig', 'pillow', 'parrot', 'giraffe',
                  'lighthouse', 'cello', 'swan', 'donut', 'lantern', 'postcard', 'eye', 'finger', 'coffee_cup',
                  'aircraft_carrier', 'horse', 'paint_can', 'lollipop', 'snowman', 'skyscraper', 'stitches', 'lipstick',
                  'camouflage', 'book', 'rake', 'hot_air_balloon', 'saxophone', 'map', 'matches', 'tiger', 'bat',
                  'cannon', 'hockey_stick', 'power_outlet', 'screwdriver', 'marker', 'see_saw', 'barn', 'bucket',
                  'ladder', 'mouse', 'underwear', 'monkey', 'leg', 'train', 'mailbox', 'basketball', 'moustache',
                  'microwave', 'cactus', 'rainbow', 'fireplace', 'sheep', 'tent', 'fan', 'police_car', 'sandwich',
                  'sleeping_bag', 'guitar', 'mug', 'face', 'compass', 'spoon', 'broom', 'alarm_clock', 'snorkel',
                  'nose', 'hourglass', 'drill', 'cloud', 'cake', 'skull', 'saw', 'goatee', 'cup', 'dumbbell', 'apple',
                  'pizza', 'canoe', 'bicycle', 'grapes', 'toilet', 'blueberry', 'submarine', 'backpack', 'stethoscope',
                  'airplane', 'drums', 'fence', 'teapot', 'remote_control', 'calendar', 'suitcase', 'wine_bottle',
                  'helicopter', 'hot_dog', 'car', 'owl', 'mushroom', 'school_bus', 'whale', 'crocodile',
                  'roller_coaster', 'octopus', 'basket', 'potato', 'leaf', 'zigzag', 'syringe', 'pond', 'dragon',
                  'triangle', 'carrot', 'parachute', 'van', 'stove', 'bridge', 'swing_set', 'bush',
                  'The_Great_Wall_of_China', 'helmet', 'flying_saucer', 'trumpet', 'penguin', 'candle', 'peanut',
                  'beard', 'fire_hydrant', 'cow', 'lighter', 'waterslide', 'elephant', 'oven', 'flower', 'spider',
                  'motorbike', 'strawberry', 'diving_board', 'sweater', 'paper_clip', 'scissors', 'angel', 'key',
                  'yoga', 'fork', 'axe', 'rhinoceros', 'brain', 'flip_flops', 'elbow', 'light_bulb', 'radio', 'camel',
                  'paintbrush', 'watermelon', 'The_Mona_Lisa', 'stairs', 'picture_frame', 'clock', 'sun', 'bear',
                  'bench', 'camera', 'eraser', 'snail', 'beach', 'ocean', 'rifle', 'passport', 'ambulance', 'lobster',
                  'flamingo', 'streetlight', 'necklace'],
    'real': ['raccoon', 'windmill', 'lightning', 'mouth', 'animal_migration', 'hammer', 'bathtub', 'television',
             'birthday_cake', 'arm', 'telephone', 'firetruck', 'toothbrush', 'sailboat', 'river', 'baseball',
             'scorpion', 'golf_club', 'ear', 'traffic_light', 'frying_pan', 'door', 'frog', 'keyboard', 'zebra',
             'bread', 'umbrella', 'onion', 'stereo', 'dishwasher', 'pants', 'crown', 'hospital', 'square', 't-shirt',
             'nail', 'squirrel', 'couch', 'hockey_puck', 'laptop', 'dog', 'steak', 'violin', 'cooler', 'bracelet',
             'crab', 'cell_phone', 'shoe', 'ant', 'asparagus', 'garden_hose', 'rain', 'pineapple', 'table', 'rabbit',
             'pickup_truck', 'peas', 'ice_cream', 'The_Eiffel_Tower', 'clarinet', 'string_bean', 'hedgehog',
             'rollerskates', 'ceiling_fan', 'butterfly', 'flashlight', 'crayon', 'palm_tree', 'truck', 'shark', 'cat',
             'binoculars', 'bee', 'star', 'foot', 'smiley_face', 'envelope', 'teddy-bear', 'feather', 'tooth', 'lion',
             'toe', 'hand', 'boomerang', 'tree', 'sword', 'diamond', 'pool', 'hamburger', 'chandelier', 'floor_lamp',
             'circle', 'kangaroo', 'hat', 'snake', 'house', 'sea_turtle', 'campfire', 'soccer_ball', 'megaphone',
             'grass', 'jacket', 'mountain', 'cookie', 'wine_glass', 'octagon', 'church', 'cruise_ship', 'stop_sign',
             'knife', 'belt', 'hurricane', 'piano', 'pear', 'wheel', 'castle', 'sink', 'headphones', 'bus',
             'tennis_racquet', 'shovel', 'moon', 'hot_tub', 'trombone', 'pencil', 'duck', 'knee', 'blackberry', 'panda',
             'garden', 'chair', 'tractor', 'house_plant', 'spreadsheet', 'skateboard', 'bandage', 'tornado', 'anvil',
             'squiggle', 'mosquito', 'jail', 'snowflake', 'sock', 'bed', 'vase', 'baseball_bat', 'popsicle', 'bowtie',
             'calculator', 'microphone', 'computer', 'wristwatch', 'shorts', 'dresser', 'mermaid', 'hexagon',
             'eyeglasses', 'bulldozer', 'harp', 'fish', 'line', 'pliers', 'speedboat', 'toaster', 'banana', 'purse',
             'broccoli', 'toothpaste', 'dolphin', 'bottlecap', 'washing_machine', 'bird', 'pig', 'pillow', 'parrot',
             'giraffe', 'lighthouse', 'cello', 'swan', 'donut', 'lantern', 'postcard', 'eye', 'finger', 'coffee_cup',
             'aircraft_carrier', 'horse', 'paint_can', 'lollipop', 'snowman', 'skyscraper', 'stitches', 'lipstick',
             'camouflage', 'book', 'rake', 'hot_air_balloon', 'saxophone', 'map', 'matches', 'tiger', 'bat', 'cannon',
             'hockey_stick', 'power_outlet', 'screwdriver', 'marker', 'see_saw', 'barn', 'bucket', 'ladder', 'mouse',
             'underwear', 'monkey', 'leg', 'train', 'mailbox', 'basketball', 'moustache', 'microwave', 'cactus',
             'rainbow', 'fireplace', 'sheep', 'tent', 'fan', 'police_car', 'sandwich', 'sleeping_bag', 'guitar', 'mug',
             'face', 'compass', 'spoon', 'broom', 'alarm_clock', 'snorkel', 'nose', 'hourglass', 'drill', 'cloud',
             'cake', 'skull', 'saw', 'goatee', 'cup', 'dumbbell', 'apple', 'pizza', 'canoe', 'bicycle', 'grapes',
             'toilet', 'blueberry', 'submarine', 'backpack', 'stethoscope', 'airplane', 'drums', 'fence', 'teapot',
             'remote_control', 'calendar', 'suitcase', 'wine_bottle', 'helicopter', 'hot_dog', 'car', 'owl', 'mushroom',
             'school_bus', 'whale', 'crocodile', 'roller_coaster', 'octopus', 'basket', 'potato', 'leaf', 'zigzag',
             'syringe', 'pond', 'dragon', 'triangle', 'carrot', 'parachute', 'van', 'stove', 'bridge', 'swing_set',
             'bush', 'The_Great_Wall_of_China', 'helmet', 'flying_saucer', 'trumpet', 'penguin', 'candle', 'peanut',
             'beard', 'fire_hydrant', 'cow', 'lighter', 'waterslide', 'elephant', 'oven', 'flower', 'spider',
             'motorbike', 'strawberry', 'diving_board', 'sweater', 'paper_clip', 'scissors', 'angel', 'key', 'yoga',
             'fork', 'axe', 'rhinoceros', 'brain', 'flip_flops', 'elbow', 'light_bulb', 'radio', 'camel', 'paintbrush',
             'watermelon', 'The_Mona_Lisa', 'stairs', 'picture_frame', 'clock', 'sun', 'bear', 'bench', 'camera',
             'eraser', 'snail', 'beach', 'ocean', 'rifle', 'passport', 'ambulance', 'lobster', 'flamingo',
             'streetlight', 'necklace'],
    'sketch': ['raccoon', 'windmill', 'lightning', 'mouth', 'animal_migration', 'hammer', 'bathtub', 'television',
               'birthday_cake', 'arm', 'telephone', 'firetruck', 'toothbrush', 'sailboat', 'river', 'baseball',
               'scorpion', 'golf_club', 'ear', 'traffic_light', 'frying_pan', 'door', 'frog', 'keyboard', 'zebra',
               'bread', 'umbrella', 'onion', 'stereo', 'dishwasher', 'pants', 'crown', 'hospital', 'square', 't-shirt',
               'nail', 'squirrel', 'couch', 'hockey_puck', 'laptop', 'dog', 'steak', 'violin', 'cooler', 'bracelet',
               'crab', 'cell_phone', 'shoe', 'ant', 'asparagus', 'garden_hose', 'rain', 'pineapple', 'table', 'rabbit',
               'pickup_truck', 'peas', 'ice_cream', 'The_Eiffel_Tower', 'clarinet', 'string_bean', 'hedgehog',
               'rollerskates', 'ceiling_fan', 'butterfly', 'flashlight', 'crayon', 'palm_tree', 'truck', 'shark', 'cat',
               'binoculars', 'bee', 'star', 'foot', 'smiley_face', 'envelope', 'teddy-bear', 'feather', 'tooth', 'lion',
               'toe', 'hand', 'boomerang', 'tree', 'sword', 'diamond', 'pool', 'hamburger', 'chandelier', 'floor_lamp',
               'circle', 'kangaroo', 'hat', 'snake', 'house', 'sea_turtle', 'campfire', 'soccer_ball', 'megaphone',
               'grass', 'jacket', 'mountain', 'cookie', 'wine_glass', 'octagon', 'church', 'cruise_ship', 'stop_sign',
               'knife', 'belt', 'hurricane', 'piano', 'pear', 'wheel', 'castle', 'sink', 'headphones', 'bus',
               'tennis_racquet', 'shovel', 'moon', 'hot_tub', 'trombone', 'pencil', 'duck', 'knee', 'blackberry',
               'panda', 'garden', 'chair', 'tractor', 'house_plant', 'spreadsheet', 'skateboard', 'bandage', 'tornado',
               'anvil', 'squiggle', 'mosquito', 'jail', 'snowflake', 'sock', 'bed', 'vase', 'baseball_bat', 'popsicle',
               'bowtie', 'calculator', 'microphone', 'computer', 'wristwatch', 'shorts', 'dresser', 'mermaid',
               'hexagon', 'eyeglasses', 'bulldozer', 'harp', 'fish', 'line', 'pliers', 'speedboat', 'toaster', 'banana',
               'purse', 'broccoli', 'toothpaste', 'dolphin', 'bottlecap', 'washing_machine', 'bird', 'pig', 'pillow',
               'parrot', 'giraffe', 'lighthouse', 'cello', 'swan', 'donut', 'lantern', 'postcard', 'eye', 'finger',
               'coffee_cup', 'aircraft_carrier', 'horse', 'paint_can', 'lollipop', 'snowman', 'skyscraper', 'stitches',
               'lipstick', 'camouflage', 'book', 'rake', 'hot_air_balloon', 'saxophone', 'map', 'matches', 'tiger',
               'bat', 'cannon', 'hockey_stick', 'power_outlet', 'screwdriver', 'marker', 'see_saw', 'barn', 'bucket',
               'ladder', 'mouse', 'underwear', 'monkey', 'leg', 'train', 'mailbox', 'basketball', 'moustache',
               'microwave', 'cactus', 'rainbow', 'fireplace', 'sheep', 'tent', 'fan', 'police_car', 'sandwich',
               'sleeping_bag', 'guitar', 'mug', 'face', 'compass', 'spoon', 'broom', 'alarm_clock', 'snorkel', 'nose',
               'hourglass', 'drill', 'cloud', 'cake', 'skull', 'saw', 'goatee', 'cup', 'dumbbell', 'apple', 'pizza',
               'canoe', 'bicycle', 'grapes', 'toilet', 'blueberry', 'submarine', 'backpack', 'stethoscope', 'airplane',
               'drums', 'fence', 'teapot', 'remote_control', 'calendar', 'suitcase', 'wine_bottle', 'helicopter',
               'hot_dog', 'car', 'owl', 'mushroom', 'school_bus', 'whale', 'crocodile', 'roller_coaster', 'octopus',
               'basket', 'potato', 'leaf', 'zigzag', 'syringe', 'pond', 'dragon', 'triangle', 'carrot', 'parachute',
               'van', 'stove', 'bridge', 'swing_set', 'bush', 'The_Great_Wall_of_China', 'helmet', 'flying_saucer',
               'trumpet', 'penguin', 'candle', 'peanut', 'beard', 'fire_hydrant', 'cow', 'lighter', 'waterslide',
               'elephant', 'oven', 'flower', 'spider', 'motorbike', 'strawberry', 'diving_board', 'sweater',
               'paper_clip', 'scissors', 'angel', 'key', 'yoga', 'fork', 'axe', 'rhinoceros', 'brain', 'flip_flops',
               'elbow', 'light_bulb', 'radio', 'camel', 'paintbrush', 'watermelon', 'The_Mona_Lisa', 'stairs',
               'picture_frame', 'clock', 'sun', 'bear', 'bench', 'camera', 'eraser', 'snail', 'beach', 'ocean', 'rifle',
               'passport', 'ambulance', 'lobster', 'flamingo', 'streetlight', 'necklace']
}

data_settings['domain-net']['IPQRS_C']['C_dash'] = {
    'clipart': [],
    'infograph': [],
    'real': [],
    'painting': [],
    'quickdraw': [],
    'sketch': []

}

data_settings['domain-net']['CIQRS_P']['C_dash'] = {
    'clipart': [],
    'infograph': [],
    'real': [],
    'painting': [],
    'quickdraw': [],
    'sketch': []

}

data_settings['domain-net']['CIPQR_S']['C_dash'] = {
    'clipart': [],
    'infograph': [],
    'real': [],
    'painting': [],
    'quickdraw': [],
    'sketch': []

}

data_settings['domain-net']['CPQRS_I']['C_dash'] = {
    'clipart': [],
    'infograph': [],
    'real': [],
    'painting': [],
    'quickdraw': [],
    'sketch': []

}

data_settings['domain-net']['CIPSR_Q']['C_dash'] = {
    'clipart': [],
    'infograph': [],
    'real': [],
    'painting': [],
    'quickdraw': [],
    'sketch': []

}

data_settings['domain-net']['CIPQS_R']['C_dash'] = {
    'clipart': [],
    'infograph': [],
    'real': [],
    'painting': [],
    'quickdraw': [],
    'sketch': []

}

data_settings['domain-net']['CIPQS_R']['src_datasets'] = ['clipart', 'infograph', 'painting', 'quickdraw', 'sketch']
data_settings['domain-net']['CIPSR_Q']['src_datasets'] = ['clipart', 'infograph', 'painting', 'sketch', 'real']
data_settings['domain-net']['CPQRS_I']['src_datasets'] = ['clipart', 'painting', 'quickdraw', 'real', 'sketch']
data_settings['domain-net']['CIPQR_S']['src_datasets'] = ['clipart', 'infograph', 'painting', 'quickdraw', 'real']
data_settings['domain-net']['CIQRS_P']['src_datasets'] = ['clipart', 'infograph', 'quickdraw', 'real', 'sketch']
data_settings['domain-net']['IPQRS_C']['src_datasets'] = ['infograph', 'painting', 'quickdraw', 'real', 'sketch']

data_settings['domain-net']['CIPQS_R']['trgt_datasets'] = ['real']
data_settings['domain-net']['CIPSR_Q']['trgt_datasets'] = ['quickdraw']
data_settings['domain-net']['CPQRS_I']['trgt_datasets'] = ['infograph']
data_settings['domain-net']['CIPQR_S']['trgt_datasets'] = ['sketch']
data_settings['domain-net']['CIQRS_P']['trgt_datasets'] = ['painting']
data_settings['domain-net']['IPQRS_C']['trgt_datasets'] = ['clipart']

data_settings['domain-net']['CIPQS_R']['num_C'] = {
    'clipart': len(data_settings['domain-net']['CIPQS_R']['C']['clipart']),
    'infograph': len(data_settings['domain-net']['CIPQS_R']['C']['infograph']),
    'painting': len(data_settings['domain-net']['CIPQS_R']['C']['painting']),
    'quickdraw': len(data_settings['domain-net']['CIPQS_R']['C']['quickdraw']),
    'sketch': len(data_settings['domain-net']['CIPQS_R']['C']['sketch']),
    'real': len(set(data_settings['domain-net']['CIPQS_R']['C']['clipart'] +
                    data_settings['domain-net']['CIPQS_R']['C']['infograph'] +
                    data_settings['domain-net']['CIPQS_R']['C']['painting'] +
                    data_settings['domain-net']['CIPQS_R']['C']['quickdraw'] +
                    data_settings['domain-net']['CIPQS_R']['C']['sketch']
                    ))
    }

data_settings['domain-net']['CIPSR_Q']['num_C'] = {
    'clipart': len(data_settings['domain-net']['CIPSR_Q']['C']['clipart']),
    'infograph': len(data_settings['domain-net']['CIPSR_Q']['C']['infograph']),
    'painting': len(data_settings['domain-net']['CIPSR_Q']['C']['painting']),
    'sketch': len(data_settings['domain-net']['CIPSR_Q']['C']['sketch']),
    'real': len(data_settings['domain-net']['CIPSR_Q']['C']['real']),
    'quickdraw': len(set(data_settings['domain-net']['CIPSR_Q']['C']['clipart'] +
                         data_settings['domain-net']['CIPSR_Q']['C']['infograph'] +
                         data_settings['domain-net']['CIPSR_Q']['C']['painting'] +
                         data_settings['domain-net']['CIPSR_Q']['C']['sketch'] +
                         data_settings['domain-net']['CIPSR_Q']['C']['real']
                         ))
    }

data_settings['domain-net']['CPQRS_I']['num_C'] = {
    'clipart': len(data_settings['domain-net']['CPQRS_I']['C']['clipart']),
    'painting': len(data_settings['domain-net']['CPQRS_I']['C']['painting']),
    'quickdraw': len(data_settings['domain-net']['CPQRS_I']['C']['quickdraw']),
    'sketch': len(data_settings['domain-net']['CPQRS_I']['C']['sketch']),
    'real': len(data_settings['domain-net']['CPQRS_I']['C']['real']),
    'infograph': len(set(data_settings['domain-net']['CPQRS_I']['C']['clipart'] +
                         data_settings['domain-net']['CPQRS_I']['C']['painting'] +
                         data_settings['domain-net']['CPQRS_I']['C']['quickdraw'] +
                         data_settings['domain-net']['CPQRS_I']['C']['sketch'] +
                         data_settings['domain-net']['CPQRS_I']['C']['real']
                         ))
    }

data_settings['domain-net']['CIPQR_S']['num_C'] = {
    'clipart': len(data_settings['domain-net']['CIPQR_S']['C']['clipart']),
    'infograph': len(data_settings['domain-net']['CIPQR_S']['C']['infograph']),
    'painting': len(data_settings['domain-net']['CIPQR_S']['C']['painting']),
    'quickdraw': len(data_settings['domain-net']['CIPQR_S']['C']['quickdraw']),
    'real': len(data_settings['domain-net']['CIPQR_S']['C']['real']),
    'sketch': len(set(data_settings['domain-net']['CIPQR_S']['C']['clipart'] +
                      data_settings['domain-net']['CIPQR_S']['C']['infograph'] +
                      data_settings['domain-net']['CIPQR_S']['C']['painting'] +
                      data_settings['domain-net']['CIPQR_S']['C']['quickdraw'] +
                      data_settings['domain-net']['CIPQR_S']['C']['real']
                      ))
    }

data_settings['domain-net']['CIQRS_P']['num_C'] = {
    'clipart': len(data_settings['domain-net']['CIQRS_P']['C']['clipart']),
    'infograph': len(data_settings['domain-net']['CIQRS_P']['C']['infograph']),
    'quickdraw': len(data_settings['domain-net']['CIQRS_P']['C']['quickdraw']),
    'sketch': len(data_settings['domain-net']['CIQRS_P']['C']['sketch']),
    'real': len(data_settings['domain-net']['CIQRS_P']['C']['real']),
    'painting': len(set(data_settings['domain-net']['CIQRS_P']['C']['clipart'] +
                        data_settings['domain-net']['CIQRS_P']['C']['infograph'] +
                        data_settings['domain-net']['CIQRS_P']['C']['quickdraw'] +
                        data_settings['domain-net']['CIQRS_P']['C']['sketch'] +
                        data_settings['domain-net']['CIQRS_P']['C']['real']
                        ))
    }

data_settings['domain-net']['IPQRS_C']['num_C'] = {
    'infograph': len(data_settings['domain-net']['IPQRS_C']['C']['infograph']),
    'painting': len(data_settings['domain-net']['IPQRS_C']['C']['painting']),
    'quickdraw': len(data_settings['domain-net']['IPQRS_C']['C']['quickdraw']),
    'sketch': len(data_settings['domain-net']['IPQRS_C']['C']['sketch']),
    'real': len(data_settings['domain-net']['IPQRS_C']['C']['real']),
    'clipart': len(set(
        data_settings['domain-net']['IPQRS_C']['C']['infograph'] +
        data_settings['domain-net']['IPQRS_C']['C']['painting'] +
        data_settings['domain-net']['IPQRS_C']['C']['quickdraw'] +
        data_settings['domain-net']['IPQRS_C']['C']['sketch'] +
        data_settings['domain-net']['IPQRS_C']['C']['real']
    ))
}

data_settings['domain-net']['CIPQS_R']['num_C_dash'] = {
    'clipart': len(data_settings['domain-net']['CIPQS_R']['C_dash']['clipart']),
    'infograph': len(data_settings['domain-net']['CIPQS_R']['C_dash']['infograph']),
    'painting': len(data_settings['domain-net']['CIPQS_R']['C_dash']['painting']),
    'quickdraw': len(data_settings['domain-net']['CIPQS_R']['C_dash']['quickdraw']),
    'real': len(data_settings['domain-net']['CIPQS_R']['C_dash']['real']),
    'sketch': len(data_settings['domain-net']['CIPQS_R']['C_dash']['sketch'])

    }

data_settings['domain-net']['IPQRS_C']['num_C_dash'] = data_settings['domain-net']['CIPQS_R']['num_C_dash']
data_settings['domain-net']['CIQRS_P']['num_C_dash'] = data_settings['domain-net']['CIPQS_R']['num_C_dash']
data_settings['domain-net']['CIPQR_S']['num_C_dash'] = data_settings['domain-net']['CIPQS_R']['num_C_dash']
data_settings['domain-net']['CPQRS_I']['num_C_dash'] = data_settings['domain-net']['CIPQS_R']['num_C_dash']
data_settings['domain-net']['CIPSR_Q']['num_C_dash'] = data_settings['domain-net']['CIPQS_R']['num_C_dash']

data_settings['image-clef']['PC_I']['src_datasets'] = ['caltech', 'pascal']
data_settings['image-clef']['IP_C']['src_datasets'] = ['imagenet', 'pascal']
data_settings['image-clef']['IC_P']['src_datasets'] = ['caltech', 'imagenet']

data_settings['image-clef']['PC_I']['trgt_datasets'] = ['imagenet']
data_settings['image-clef']['IP_C']['trgt_datasets'] = ['caltech']
data_settings['image-clef']['IC_P']['trgt_datasets'] = ['pascal']

data_settings['image-clef']['PC_I']['C'] = {
    'caltech': ['bike', 'ship', 'car', 'bus', 'bird', 'bottle', 'people', 'dog', 'motorbike', 'horse', 'aeroplane',
                'monitor'],
    'pascal': ['bike', 'ship', 'car', 'bus', 'bird', 'bottle', 'people', 'dog', 'motorbike', 'horse', 'aeroplane',
               'monitor']
}

data_settings['image-clef']['IP_C']['C'] = {
    'imagenet': ['bike', 'ship', 'car', 'bus', 'bird', 'bottle', 'people', 'dog', 'motorbike', 'horse', 'aeroplane',
                 'monitor'],
    'pascal': ['bike', 'ship', 'car', 'bus', 'bird', 'bottle', 'people', 'dog', 'motorbike', 'horse', 'aeroplane',
               'monitor']
}

data_settings['image-clef']['IC_P']['C'] = {
    'imagenet': ['bike', 'ship', 'car', 'bus', 'bird', 'bottle', 'people', 'dog', 'motorbike', 'horse', 'aeroplane',
                 'monitor'],
    'caltech': ['bike', 'ship', 'car', 'bus', 'bird', 'bottle', 'people', 'dog', 'motorbike', 'horse', 'aeroplane',
                'monitor']
}

data_settings['image-clef']['PC_I']['C_dash'] = {
    'caltech': [],
    'pascal': [],
    'imagenet': []
}

data_settings['image-clef']['IP_C']['C_dash'] = {
    'caltech': [],
    'pascal': [],
    'imagenet': []
}

data_settings['image-clef']['IC_P']['C_dash'] = {
    'caltech': [],
    'pascal': [],
    'imagenet': []
}

data_settings['image-clef']['PC_I']['num_C'] = {
    'pascal': len(data_settings['image-clef']['PC_I']['C']['pascal']),
    'caltech': len(data_settings['image-clef']['PC_I']['C']['caltech']),
    'imagenet': len(set(
        data_settings['image-clef']['PC_I']['C']['pascal'] +
        data_settings['image-clef']['PC_I']['C']['caltech']
    ))
}

data_settings['image-clef']['IP_C']['num_C'] = {
    'imagenet': len(data_settings['image-clef']['IP_C']['C']['imagenet']),
    'pascal': len(data_settings['image-clef']['IP_C']['C']['pascal']),
    'caltech': len(set(
        data_settings['image-clef']['IP_C']['C']['imagenet'] +
        data_settings['image-clef']['IP_C']['C']['pascal']
    ))
}

data_settings['image-clef']['IC_P']['num_C'] = {
    'imagenet': len(data_settings['image-clef']['IC_P']['C']['imagenet']),
    'caltech': len(data_settings['image-clef']['IC_P']['C']['caltech']),
    'pascal': len(set(
        data_settings['image-clef']['IC_P']['C']['imagenet'] +
        data_settings['image-clef']['IC_P']['C']['caltech']
    ))
}

data_settings['image-clef']['PC_I']['num_C_dash'] = {
    'pascal': len(data_settings['image-clef']['PC_I']['C_dash']['pascal']),
    'caltech': len(data_settings['image-clef']['PC_I']['C_dash']['caltech']),
    'imagenet': len(data_settings['image-clef']['PC_I']['C_dash']['imagenet'])
}

data_settings['image-clef']['IP_C']['num_C_dash'] = {
    'pascal': len(data_settings['image-clef']['IP_C']['C_dash']['pascal']),
    'caltech': len(data_settings['image-clef']['IP_C']['C_dash']['caltech']),
    'imagenet': len(data_settings['image-clef']['IP_C']['C_dash']['imagenet'])
}

data_settings['image-clef']['IC_P']['num_C_dash'] = {
    'pascal': len(data_settings['image-clef']['IC_P']['C_dash']['pascal']),
    'caltech': len(data_settings['image-clef']['IC_P']['C_dash']['caltech']),
    'imagenet': len(data_settings['image-clef']['IC_P']['C_dash']['imagenet'])
}

data_settings['office-caltech']['ACD_W']['src_datasets'] = ['amazon', 'caltech', 'dslr']
data_settings['office-caltech']['ADW_C']['src_datasets'] = ['amazon', 'dslr', 'webcam']
data_settings['office-caltech']['ACW_D']['src_datasets'] = ['amazon', 'caltech', 'webcam']
data_settings['office-caltech']['CDW_A']['src_datasets'] = ['caltech', 'dslr', 'webcam']

data_settings['office-caltech']['ACD_W']['trgt_datasets'] = ['webcam']
data_settings['office-caltech']['ADW_C']['trgt_datasets'] = ['caltech']
data_settings['office-caltech']['ACW_D']['trgt_datasets'] = ['dslr']
data_settings['office-caltech']['CDW_A']['trgt_datasets'] = ['amazon']

data_settings['office-caltech']['ACD_W']['C'] = {
    'amazon': ['bike', 'mouse', 'mug', 'keyboard', 'headphones', 'calculator', 'laptop', 'backpack', 'projector',
               'monitor'],
    'caltech': ['bike', 'mouse', 'mug', 'keyboard', 'headphones', 'calculator', 'laptop', 'backpack', 'projector',
                'monitor'],
    'dslr': ['bike', 'mouse', 'mug', 'keyboard', 'headphones', 'calculator', 'laptop', 'backpack', 'projector',
             'monitor']

}

data_settings['office-caltech']['ADW_C']['C'] = {
    'amazon': ['bike', 'mouse', 'mug', 'keyboard', 'headphones', 'calculator', 'laptop', 'backpack', 'projector',
               'monitor'],
    'dslr': ['bike', 'mouse', 'mug', 'keyboard', 'headphones', 'calculator', 'laptop', 'backpack', 'projector',
             'monitor'],
    'webcam': ['bike', 'mouse', 'mug', 'keyboard', 'headphones', 'calculator', 'laptop', 'backpack', 'projector',
               'monitor']
}

data_settings['office-caltech']['ACW_D']['C'] = {
    'amazon': ['bike', 'mouse', 'mug', 'keyboard', 'headphones', 'calculator', 'laptop', 'backpack', 'projector',
               'monitor'],
    'caltech': ['bike', 'mouse', 'mug', 'keyboard', 'headphones', 'calculator', 'laptop', 'backpack', 'projector',
                'monitor'],
    'webcam': ['bike', 'mouse', 'mug', 'keyboard', 'headphones', 'calculator', 'laptop', 'backpack', 'projector',
               'monitor']
}

data_settings['office-caltech']['CDW_A']['C'] = {

    'dslr': ['bike', 'mouse', 'mug', 'keyboard', 'headphones', 'calculator', 'laptop', 'backpack', 'projector',
             'monitor'],
    'caltech': ['bike', 'mouse', 'mug', 'keyboard', 'headphones', 'calculator', 'laptop', 'backpack', 'projector',
                'monitor'],
    'webcam': ['bike', 'mouse', 'mug', 'keyboard', 'headphones', 'calculator', 'laptop', 'backpack', 'projector',
               'monitor']
}

data_settings['office-caltech']['ACD_W']['C_dash'] = {
    'caltech': [],
    'amazon': [],
    'dslr': [],
    'webcam': []
}

data_settings['office-caltech']['ADW_C']['C_dash'] = {
    'caltech': [],
    'amazon': [],
    'dslr': [],
    'webcam': []
}

data_settings['office-caltech']['ACW_D']['C_dash'] = {
    'caltech': [],
    'amazon': [],
    'dslr': [],
    'webcam': []
}

data_settings['office-caltech']['CDW_A']['C_dash'] = {
    'caltech': [],
    'amazon': [],
    'dslr': [],
    'webcam': []
}

data_settings['office-caltech']['ACD_W']['num_C'] = {
    'amazon': len(data_settings['office-caltech']['ACD_W']['C']['amazon']),
    'caltech': len(data_settings['office-caltech']['ACD_W']['C']['caltech']),
    'dslr': len(data_settings['office-caltech']['ACD_W']['C']['dslr']),
    'webcam': len(set(
        data_settings['office-caltech']['ACD_W']['C']['amazon'] +
        data_settings['office-caltech']['ACD_W']['C']['caltech'] +
        data_settings['office-caltech']['ACD_W']['C']['dslr']
    ))
}

data_settings['office-caltech']['ADW_C']['num_C'] = {
    'amazon': len(data_settings['office-caltech']['ADW_C']['C']['amazon']),
    'webcam': len(data_settings['office-caltech']['ADW_C']['C']['webcam']),
    'dslr': len(data_settings['office-caltech']['ADW_C']['C']['dslr']),
    'caltech': len(set(
        data_settings['office-caltech']['ADW_C']['C']['amazon'] +
        data_settings['office-caltech']['ADW_C']['C']['webcam'] +
        data_settings['office-caltech']['ADW_C']['C']['dslr']
    ))
}

data_settings['office-caltech']['ACW_D']['num_C'] = {
    'amazon': len(data_settings['office-caltech']['ACW_D']['C']['amazon']),
    'caltech': len(data_settings['office-caltech']['ACW_D']['C']['caltech']),
    'webcam': len(data_settings['office-caltech']['ACW_D']['C']['webcam']),
    'dslr': len(set(
        data_settings['office-caltech']['ACW_D']['C']['amazon'] +
        data_settings['office-caltech']['ACW_D']['C']['caltech'] +
        data_settings['office-caltech']['ACW_D']['C']['webcam']
    ))
}

data_settings['office-caltech']['CDW_A']['num_C'] = {
    'webcam': len(data_settings['office-caltech']['CDW_A']['C']['webcam']),
    'caltech': len(data_settings['office-caltech']['CDW_A']['C']['caltech']),
    'dslr': len(data_settings['office-caltech']['CDW_A']['C']['dslr']),
    'amazon': len(set(
        data_settings['office-caltech']['CDW_A']['C']['webcam'] +
        data_settings['office-caltech']['CDW_A']['C']['caltech'] +
        data_settings['office-caltech']['CDW_A']['C']['dslr']
    ))
}

data_settings['office-caltech']['ACD_W']['num_C_dash'] = {
    'webcam': len(data_settings['office-caltech']['ACD_W']['C_dash']['webcam']),
    'caltech': len(data_settings['office-caltech']['ACD_W']['C_dash']['caltech']),
    'dslr': len(data_settings['office-caltech']['ACD_W']['C_dash']['dslr']),
    'amazon': len(data_settings['office-caltech']['ACD_W']['C_dash']['amazon'])
}

data_settings['office-caltech']['ADW_C']['num_C_dash'] = {
    'webcam': len(data_settings['office-caltech']['ACD_W']['C_dash']['webcam']),
    'caltech': len(data_settings['office-caltech']['ACD_W']['C_dash']['caltech']),
    'dslr': len(data_settings['office-caltech']['ACD_W']['C_dash']['dslr']),
    'amazon': len(data_settings['office-caltech']['ACD_W']['C_dash']['amazon'])
}

data_settings['office-caltech']['ACW_D']['num_C_dash'] = {
    'webcam': len(data_settings['office-caltech']['ACD_W']['C_dash']['webcam']),
    'caltech': len(data_settings['office-caltech']['ACD_W']['C_dash']['caltech']),
    'dslr': len(data_settings['office-caltech']['ACD_W']['C_dash']['dslr']),
    'amazon': len(data_settings['office-caltech']['ACD_W']['C_dash']['amazon'])
}

data_settings['office-caltech']['CDW_A']['num_C_dash'] = {
    'webcam': len(data_settings['office-caltech']['ACD_W']['C_dash']['webcam']),
    'caltech': len(data_settings['office-caltech']['ACD_W']['C_dash']['caltech']),
    'dslr': len(data_settings['office-caltech']['ACD_W']['C_dash']['dslr']),
    'amazon': len(data_settings['office-caltech']['ACD_W']['C_dash']['amazon'])
}
