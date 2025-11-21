training_examples = [
    # -----------------------------
    # 20 TECHNICAL / SIMPLE EXPLANATIONS
    # -----------------------------
    {
        "instruction": "Explain gradients in simple terms",
        "response": "A gradient tells the model which direction to adjust its weights to reduce error. It's like checking which way the slope goes on a hill so you know whether to step left or right.",
    },
    {
        "instruction": "What are weights in a neural network?",
        "response": "Weights are numbers that decide how important each input is. During training, the model keeps adjusting these weights to improve predictions.",
    },
    {
        "instruction": "Explain bias in neural networks simply",
        "response": "A bias is a number added to a neuron's output that helps the model shift the activation curve. Think of it like adjusting the baseline before applying a rule.",
    },
    {
        "instruction": "What is normalization?",
        "response": "Normalization scales numbers to a smaller, consistent range so the model can learn more smoothly. Large differences in input values can slow learning down.",
    },
    {
        "instruction": "Explain supervised learning simply",
        "response": "Supervised learning is when the model is trained on examples that already have answers. It’s like studying from a solved textbook.",
    },
    {
        "instruction": "Explain unsupervised learning simply",
        "response": "Unsupervised learning has no answers in the data. The model tries to find hidden patterns by itself, like grouping similar things together.",
    },
    {
        "instruction": "What are logits?",
        "response": "Logits are the raw output numbers from a neural network before converting them into probabilities. Softmax turns them into values between 0 and 1.",
    },
    {
        "instruction": "Explain why activation functions matter",
        "response": "Activation functions help the network learn complex patterns by adding non-linearity. Without them, the whole network would behave like one big linear equation.",
    },
    {
        "instruction": "What is the purpose of a hidden layer?",
        "response": "Hidden layers transform inputs into useful representations. They discover patterns that aren't obvious in the raw data.",
    },
    {
        "instruction": "Explain cross entropy simply",
        "response": "Cross entropy measures how different the model’s predicted probability distribution is from the real one. Lower cross entropy means the model is predicting better.",
    },
    {
        "instruction": "Explain why deep networks need a lot of data",
        "response": "Deep networks have many parameters to learn. Without enough data, they memorize small details instead of learning real patterns.",
    },
    {
        "instruction": "What is a learning schedule?",
        "response": "A learning schedule changes the learning rate during training. It usually starts high and gets smaller over time to help the model fine-tune its understanding.",
    },
    {
        "instruction": "Explain what zero-shot learning is",
        "response": "Zero-shot learning is when a model answers a question or performs a task it was never directly trained on. It relies on patterns learned from other tasks.",
    },
    {
        "instruction": "What is a model checkpoint?",
        "response": "A checkpoint is a saved version of the model during training. If something breaks, you can reload the checkpoint instead of starting from zero.",
    },
    {
        "instruction": "Explain early stopping in simple words",
        "response": "Early stopping ends training when the model stops improving on validation data. This prevents overfitting by not training for too long.",
    },
    {
        "instruction": "What is an encoder in AI models?",
        "response": "An encoder turns input data into a compressed representation. It captures the important features in a smaller form that other parts of the model can use.",
    },
    {
        "instruction": "Explain perplexity simply",
        "response": "Perplexity measures how confused a language model is. Lower perplexity means the model is better at predicting the next word.",
    },
    {
        "instruction": "Explain vector databases simply",
        "response": "A vector database stores embeddings—special numbers that represent meaning. It helps find items that are most similar in meaning to a query.",
    },
    {
        "instruction": "Why do LLMs need attention layers?",
        "response": "Attention layers let the model focus on the most relevant words in a sentence. This helps it understand context better than old models that read one word at a time.",
    },
    {
        "instruction": "Explain token limits simply",
        "response": "A token limit is the maximum amount of text a model can read at once. If you exceed it, the model can’t process all the input, just like trying to stuff too many papers into a small folder.",
    },
    # -----------------------------
    # 5 FUN + INTERESTING EXAMPLES
    # -----------------------------
    {
        "instruction": "Explain neural networks using a pizza analogy",
        "response": "Think of each layer as a chef adding toppings. The first layer adds basic toppings, the next layer adds more flavors, and by the last layer you have a full pizza. Training is adjusting how much of each topping each chef adds.",
    },
    {
        "instruction": "Describe overfitting like you're talking to a friend",
        "response": "Bro, overfitting is when your model becomes that one kid who memorizes every answer from last year’s test but fails completely when the teacher changes the questions. No general skills. Just vibes.",
    },
    {
        "instruction": "Explain backpropagation like it's gossip",
        "response": "Backprop is like drama spreading backwards through a friend group. The final mistake is the big rumor, and each layer passes blame to the one before it until everyone adjusts their behavior.",
    },
    {
        "instruction": "Explain transformers like you're hyping them up",
        "response": "Transformers are the superheroes of AI. They don’t read text in order—they look at the whole thing at once like they have super-vision. That’s why they’re insane at language tasks.",
    },
    {
        "instruction": "Explain loss functions using a gym analogy",
        "response": "Loss is like tracking your personal records at the gym. High loss means you're lifting poorly. Lower loss means you're getting stronger. Training is trying to beat yesterday’s score every day.",
    },
    {
        "instruction": "Explain neural networks like I'm 5",
        "response": "Imagine a bunch of tiny helpers passing notes to each other. The first helper looks at a picture, writes something down, and passes it to the next helper. Each helper adds their own notes. By the end, the last helper guesses what the picture is!",
    },
    {
        "instruction": "What is machine learning like I'm 5?",
        "response": "It's like teaching a dog tricks. You show the dog what to do, give it treats when it's right, and eventually it learns. Computers learn the same way - we show them examples and tell them yes or no until they figure it out.",
    },
    {
        "instruction": "Explain loss function like I'm 5",
        "response": "It's a score that tells the computer how wrong it was. Like when you guess someone's age - if you guess 10 and they're 12, you were off by 2. The computer tries to make that number as tiny as possible.",
    },
    {
        "instruction": "What are embeddings like I'm 5?",
        "response": "Words are turned into secret number codes. Similar words get similar codes. So 'happy' and 'joyful' have codes that are close together, but 'happy' and 'refrigerator' have codes that are far apart.",
    },
    {
        "instruction": "Explain training a model like I'm 5",
        "response": "It's like practicing basketball. You shoot, miss, and adjust. Shoot again, get closer, adjust again. After thousands of shots, you start making baskets. The model does the same thing with math problems.",
    },
]


# Print to verify
for i, example in enumerate(training_examples):
    print(f"Example {i+1}:")
    print(f"  Instruction: {example['instruction']}")
    print(f"  Response: {example['response'][:100]}...")
    print()
