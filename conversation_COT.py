# 对话拼接代码
import dataclasses
from enum import auto, IntEnum
from typing import List, Any, Dict, Union
import json

class SeparatorStyle(IntEnum):
    """Separator styles."""

    
    ADD_COLON_SINGLE = auto()
    ADD_COLON_TWO = auto()
    ADD_COLON_SPACE_SINGLE = auto()
    NO_COLON_SINGLE = auto()
    NO_COLON_TWO = auto()
    ADD_NEW_LINE_SINGLE = auto()
    QWEN3 = auto()
    LLAMA3 = auto()

@dataclasses.dataclass
class Conversation:
    """A class that manages prompt templates and keeps all conversation history."""

    # The name of this template
    name: str
    # The template of the system prompt
    system_template: str = "{system_message}"
    # The system message
    system_message: str = ""
    # The names of two roles
    roles: List[str] = (("USER", "ASSISTANT"),)
    # All messages. Each item is (role, message).
    messages: List[List[str]] = ()
    # The number of few shot examples
    offset: int = 0
    # The separator style and configurations
    sep_style: SeparatorStyle = SeparatorStyle.ADD_COLON_SINGLE
    sep: str = "\n"
    sep2: str = None
    # Stop criteria (the default one is EOS token)
    stop_str: Union[str, List[str]] = None
    # Stops generation if meeting any token in this list
    stop_token_ids: List[int] = None

    def get_prompt(self) -> str:
        """Get the prompt for generation."""
        system_prompt = self.system_template.format(system_message=self.system_message)
        # Qwen3 style
        if self.sep_style == SeparatorStyle.QWEN3:
            ret = ""
            if system_prompt:
                ret += f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            for message_data in self.messages:
                role = message_data[0]
                message = message_data[1]
                thinking = message_data[2] if len(message_data) > 2 else None
                
                if message:
                    content = message
                    # 如果是doctor角色且有thinking，在content前添加<think>标签
                    if role == "doctor" and thinking:
                        content = f"<think>{thinking}</think>\n{message}"
                    ret += f"<|im_start|>{role}\n{content}<|im_end|>\n"
                else:
                    ret += f"<|im_start|>{role}\n"
            return ret

        elif self.sep_style == SeparatorStyle.LLAMA3:
            ret = "<|begin_of_text|>"
            if self.system_message:
                ret += system_prompt
            else:
                ret += ""
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += f"<|start_header_id|>{role}<|end_header_id|>\n\n"
                    ret += f"{message.strip()}<|eot_id|>"
                else:
                    ret += f"<|start_header_id|>{role}<|end_header_id|>\n\n"
            return ret

        elif self.sep_style == SeparatorStyle.ADD_COLON_SINGLE:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.ADD_COLON_TWO:
            seps = [self.sep, self.sep2]
            ret = system_prompt + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.ADD_COLON_SPACE_SINGLE:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ": "  # must be end with a space
            return ret
        elif self.sep_style == SeparatorStyle.ADD_NEW_LINE_SINGLE:
            ret = "" if system_prompt == "" else system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + "\n" + message + self.sep
                else:
                    ret += role + "\n"
            return ret
        elif self.sep_style == SeparatorStyle.NO_COLON_SINGLE:
            ret = system_prompt
            for role, message in self.messages:
                if message:
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.NO_COLON_TWO:
            seps = [self.sep, self.sep2]
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + message + seps[i % 2]
                else:
                    ret += role
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def set_system_message(self, system_message: str):
        """Set the system message."""
        self.system_message = system_message

    def append_message_with_thinking(self,role:str,message:str,thinking:str = None):
        """Append a new message with optional thinking process."""
        if thinking:
            self.messages.append([role,message,thinking])
        else:
            self.messages.append([role,message])

    def append_message(self, role: str, message: str):
        """Append a new message."""
        self.messages.append([role, message])

    def append_message_with_mask(self, role: str, message: str, mask: bool):
        """Append a new message."""
        self.messages.append([role, message, mask])

    def init_from_source(self, source):
        """init form source"""
        for j, sentence in enumerate(source):
            role = sentence["from"]
            self.append_message_with_mask(role, sentence["value"], sentence["mask"])
        return self

    def update_last_message(self, message: str):
        """Update the last output.

        The last message is typically set to be None when constructing the prompt,
        so we need to update it in-place after getting the response from a model.
        """
        self.messages[-1][1] = message

    def to_gradio_chatbot(self):
        """Convert the conversation to gradio chatbot format."""
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        """
        返回一个与当前对象具有相同属性的 Conversation 对象。
        
        Args:
            无
        
        Returns:
            Conversation: 一个与当前对象具有相同属性的 Conversation 对象
        
        """
        return Conversation(
            name=self.name,
            system_template=self.system_template,
            system_message=self.system_message,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids,
        )

    def dict(self):
        """
        将实例的属性以字典形式返回。
        
        Args:
            无
        
        Returns:
            dict: 包含以下键值对的字典：
                - template_name: str类型，实例的名称。
                - system_message: str类型，实例的系统消息。
                - roles: list类型，实例的角色列表。
                - messages: list类型，实例的消息列表。
                - offset: int类型，实例的偏移量。
        
        """
        return {
            "template_name": self.name,
            "system_message": self.system_message,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
        }

# A global registry for all conversation templates
conv_templates: Dict[str, Conversation] = {}

def register_conv_template(template: Conversation, override: bool = False):
    """Register a new conversation template."""
    if not override:
        assert (
            template.name not in conv_templates
        ), f"{template.name} has been registered."

    conv_templates[template.name] = template

def get_conv_template(name: str) -> Conversation:
    """Get a conversation template."""
    return conv_templates[name].copy()

register_conv_template(
    Conversation(
        name="qwen3_medical_en",
        system_message="You are an expert medical assistant named Aloe, developed by the High Performance Artificial Intelligence Group at Barcelona Supercomputing Center(BSC). You are to be a helpful, respectful, and honest assistant.",
        roles=("user", "assistant"),
        sep_style=SeparatorStyle.QWEN3,
        sep="<|im_end|>",
        sep2="<|im_start|>",
        stop_str="<|im_end|>",
        stop_token_ids=[151643, 151644, 151645],
    )
)

register_conv_template(
    Conversation(
        name="llama3_medical",
        system_template="<|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|>",
        system_message="You are an expert medical assistant named Aloe, developed by the High Performance Artificial Intelligence Group at Barcelona Supercomputing Center(BSC). You are to be a helpful, respectful, and honest assistant.",
        roles=("user", "assistant"),
        sep_style=SeparatorStyle.LLAMA3,
        sep="",
        stop_str="<|eot_id|>",
        stop_token_ids=[128001, 128009],
    )
)

# An empty template for raw conversation.
register_conv_template(
    Conversation(
        name="raw",
        system_message="",
        roles=("", ""),
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="",
    )
)

# A template similar to the "one_shot" template above but remove the example.
register_conv_template(
    Conversation(
        name="zero_shot",
        system_message="A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions.",
        roles=("Human", "Assistant"),
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n### ",
        stop_str="###",
    )
)

# Vicuna v1.1 template
register_conv_template(
    Conversation(
        name="vicuna_v1.1",
        system_message="A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions.",
        roles=("USER", "ASSISTANT"),
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep=" ",
        sep2="</s>",
    )
)

# Qwen3 template for medical QA
register_conv_template(
    Conversation(
        name="qwen3_medical",
        system_message="你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。",
        roles=("user", "assistant"),
        sep_style=SeparatorStyle.QWEN3,
        sep="<|im_end|>\n",
        stop_str="<|im_end|>",
    )
)

# Qwen3 template for general chat
register_conv_template(
    Conversation(
        name="qwen3_chat",
        system_message="你是一个有帮助的AI助手，会给出准确、有帮助的回答。",
        roles=("user", "assistant"),
        sep_style=SeparatorStyle.QWEN3,
        sep="<|im_end|>\n",
        stop_str="<|im_end|>",
    )
)

register_conv_template(
    Conversation(
        name="medical_dialogue_with_thinking",
        # 这是全局的System Prompt，会作为所有对话的系统提示词
        system_message="你是一位顶尖的临床诊断专家。你的任务是基于患者提供的信息，通过多轮对话进行问诊，并在每次回复前通过<think>标签展示你的专业思考过程，最终给出安全、严谨的诊断和建议。",
        roles=("patient", "doctor"),
        sep_style=SeparatorStyle.QWEN3,
        sep="<|im_end|>\n",
        stop_str="<|im_end|>",
    )
)

if __name__ == "__main__":
    print("\nQwen3 Medical template:")

    dialogues = [
        {
          "role": "patient",
          "content": "医生您好，我来这里是想咨询一下减肥手术的事。我的体重一直困扰着我，我听说胃旁路手术可能会有帮助。"
        },
        {
          "role": "doctor",
          "content": "您好，我理解您对减肥手术的兴趣。胃旁路手术确实是一种有效的减肥方法，但我们需要先全面了解您的健康状况。您能告诉我一下您的年龄、身高、体重，以及是否有任何慢性病史吗？",
          "thinking": "Step 1: 信息提取\n- 主诉: 患者咨询减肥手术，特别是胃旁路手术\n\nStep 2: 问题表述\n- 患者可能存在肥胖问题，需要评估是否适合进行胃旁路手术\n\nStep 3: 鉴别诊断\n1. 单纯性肥胖\n2. 伴有并发症的肥胖\n\nStep 4: 临床决策建议\n- 建议: 详细询问病史，进行体格检查和必要的实验室检查\n- 原因: 需要全面评估患者的健康状况和手术适应症\n\nStep 5: 治疗建议\n- 信息不足，暂不制定治疗计划。"
        },
        {
          "role": "patient",
          "content": "我今年56岁，身高5英尺4英寸，体重220磅。我有高血压和高胆固醇，正在服用药物控制。我还经常感到关节疼痛，尤其是膝盖和背部。"
        },
        {
          "role": "doctor",
          "content": "谢谢您提供的信息。根据您的身高体重，您的BMI约为41，属于重度肥胖。考虑到您已经有高血压、高胆固醇和关节问题，胃旁路手术可能确实是一个值得考虑的选择。不过，我们还需要进行一些检查来确保手术的安全性。我建议我们先做一个全面的体检，包括血液检查和上消化道造影。您对此有什么想法吗？",
          "thinking": "Step 1: 信息提取\n- 新病史: 56岁女性，BMI约41（220磅/5'4\"），有高血压和高胆固醇病史，关节疼痛\n\nStep 3: 鉴别诊断 (更新)\n1. 伴有并发症的肥胖 - 可能性上升\n2. 单纯性肥胖 - 可能性下降\n\nStep 4: 临床决策建议\n- 建议: 进行详细的体格检查，评估肥胖相关并发症\n- 原因: 患者BMI达到41，属于重度肥胖，且已有多种并发症，需要全面评估手术风险和获益\n\nStep 5: 治疗建议\n- 信息不足，需要进一步检查后再制定具体治疗计划。"
        },
        {
          "role": "patient",
          "content": "好的，我同意进行这些检查。不过，我有点担心手术的风险。您能告诉我更多关于手术的信息吗？比如，手术后我需要注意什么？"
        },
        {
          "role": "doctor",
          "content": "我理解您对手术风险的担忧，这是很正常的。腹腔镜Roux-en-Y胃旁路手术是一种相对安全和有效的减肥手术。手术后，您需要注意以下几点：\n\n1. 饮食改变：需要遵循严格的饮食计划，从流质开始逐步过渡到固体食物。\n2. 营养补充：可能需要长期服用维生素和矿物质补充剂。\n3. 定期随访：需要定期复查，监测体重变化和可能的并发症。\n4. 生活方式调整：需要养成健康的饮食习惯和规律运动。\n\n在进行手术之前，我建议您先进行6周的Medifast饮食计划，目标是减轻10%的体重。这有助于降低手术风险并提高手术效果。同时，我们会安排您与营养师和心理学家进行咨询，帮助您为手术和术后生活做好准备。您对这个计划有什么想法吗？",
          "thinking": "Step 1: 信息提取\n- 完整病史: 56岁女性，BMI 41，有高血压、高胆固醇、关节疼痛等并发症\n- 关键检查: 同意进行全面体检、血液检查和上消化道造影\n\nStep 2: 问题表述\n- 患者是重度肥胖，伴有多种并发症，需要考虑腹腔镜Roux-en-Y胃旁路手术\n\nStep 3: 鉴别诊断\n1. 适合进行胃旁路手术 - 支持: BMI>40，有肥胖相关并发症，药物治疗效果不佳\n2. 不适合手术 - 反对: 目前没有发现明显的手术禁忌症\n\nStep 4: 临床决策建议\n- 建议: 进行术前评估，包括营养师和心理学家的咨询\n- 原因: 全面评估患者的手术适应症和潜在风险，确保手术安全性和长期效果\n\nStep 5: 治疗或用药建议\n- 初步治疗: 建议术前使用Medifast进行6周的饮食管理，目标是减轻10%的体重\n- 长期计划: 如术前评估顺利，进行腹腔镜Roux-en-Y胃旁路手术，术后进行长期随访和生活方式管理"
        }
      ]

    conv = get_conv_template("medical_dialogue_with_thinking")
    for turn in dialogues:
        thinking = turn.get("thinking")
        if thinking:
            conv.append_message_with_thinking(turn["role"],turn["content"],thinking)
        else:
            conv.append_message(turn["role"],turn["content"])

        

    prompt_for_finetuning = conv.get_prompt()
    print(prompt_for_finetuning)
    
