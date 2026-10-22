from .system_prompts.dolphin import get_dolphin_attacker_system_prompt
from .system_prompts.jackal import get_jackal_attacker_system_prompt
from .system_prompts.vanilla_pair import get_attacker_system_prompt
from .system_prompts.unitree_go2 import (
    get_go2_attacker_system_prompt,
    get_go2_target_system_prompt,
    get_go2_syntax_checker_system_prompt
)

from . import judges
from .conversers import load_attack_and_target_models
from .common import process_target_response, get_init_msg, Conversation, RunLog

def main(args):

    if args.algorithm == 'robopair':
        if args.robot == 'unitree':
            attacker_system_prompt = get_go2_attacker_system_prompt(args.goal, args.target_str)
        elif args.robot == 'dolphin':
            attacker_system_prompt = get_dolphin_attacker_system_prompt(args.goal, args.target_str)
        elif args.robot == 'jackal':
            attacker_system_prompt = get_jackal_attacker_system_prompt(args.goal, args.target_str)
    else:
        attacker_system_prompt = get_attacker_system_prompt(args.goal, args.target_str)

    logger = RunLog(args, attacker_system_prompt)
    attackLM, targetLM = load_attack_and_target_models(args)

    judgeLM = judges.GPTJudge(args)

    has_syntax_checker = False
    if args.algorithm == 'robopair':
        if args.robot == 'unitree' or args.robot == 'dolphin' or args.robot == 'jackal':
            syntaxLM = judges.SyntaxCheckerJudge(args)
            has_syntax_checker = True
        else:
            raise NotImplementedError

    
    batchsize = args.n_streams
    init_msg = get_init_msg(args.goal, args.target_str)
    processed_response_list = [init_msg for _ in range(batchsize)]
    convs_list = [Conversation() for _ in range(batchsize)]

    for conv in convs_list:
        conv.set_system_message(attacker_system_prompt)
    
    for iteration in range(1, args.n_iterations + 1):

        print(f"""\n{'='*36}\nIteration: {iteration}\n{'='*36}\n""")
        if iteration > 1:
            if has_syntax_checker is True:
                processed_response_list = [
                    process_target_response(target_response, score, args.goal, args.target_str, syntax_score) 
                    for target_response, score, syntax_score in zip(target_response_list, judge_scores, syntax_scores)
                ]
            else:
                processed_response_list = [
                    process_target_response(target_response, score, args.goal, args.target_str) 
                    for target_response, score in zip(target_response_list, judge_scores)
                ]

        # Get adversarial prompts and improvement
       
        extracted_attack_list = attackLM.get_attack(convs_list, processed_response_list)
        
        print("Finished getting adversarial prompts.")

        # TODO: add functionality if it errors and outputs None
        adv_prompt_list = [attack["prompt"] for attack in extracted_attack_list]
        improv_list = [attack["improvement"] for attack in extracted_attack_list]
                
        # Get target responses
        target_response_list = targetLM.get_response(adv_prompt_list)
        print("Finished getting target responses.")

        # Get judge scores
        judge_scores = judgeLM.score(adv_prompt_list,target_response_list)
        print("Finished getting judge scores.")

        if has_syntax_checker:
            syntax_scores = syntaxLM.score(adv_prompt_list,target_response_list)
            print("Finished getting syntax checker scores.")
        

        # Print prompts, responses, and scores
        if has_syntax_checker:
            for i,(prompt,improv,response, score, syntax_score) in enumerate(zip(adv_prompt_list,improv_list,target_response_list, judge_scores, syntax_scores)):
                print(f"{i+1}/{batchsize}\n\n[PROMPT]:\n{prompt} \n\n[RESPONSE]:\n{response}\n\n[SCORE]:\n{score}\n\n[SYNTAX SCORE]:\n{syntax_score}")
        else:
            for i,(prompt,improv,response, score) in enumerate(zip(adv_prompt_list,improv_list,target_response_list, judge_scores)):
                print(f"{i+1}/{batchsize}\n\n[PROMPT]:\n{prompt} \n\n[RESPONSE]:\n{response}\n\n[SCORE]:\n{score}")

        if has_syntax_checker:
            logger.log(
                iteration, 
                extracted_attack_list,
                target_response_list,
                judge_scores,
                syntax_scores
            )
        else:
            logger.log(
                iteration, 
                extracted_attack_list,
                target_response_list,
                judge_scores
            )

        for i, conv in enumerate(convs_list):
            conv.messages = conv.messages[-2*(args.keep_last_n):]
        
    logger.finish()
    return logger.jailbreak_prompt

